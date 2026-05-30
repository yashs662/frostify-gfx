//! Modal overlay (popup / dialog) — a self-contained, reusable widget.
//!
//! Why this exists: a modal is more than "a node positioned on top". It
//! has to (1) render above everything, (2) **block all input** to the
//! content beneath it — hover, click, scroll — not just capture its own
//! clicks, (3) animate in and out, and (4) cost nothing when closed. Hand
//! -rolling that per app is the "tape and glue" trap: you end up wiring a
//! scrim node, a stable opacity `Signal`, a `Timeline` tween key, and the
//! open/close/dismiss plumbing by hand at every call site, and it's easy
//! to get the input-blocking or the animation lifecycle subtly wrong.
//!
//! [`Overlay`] owns all of that. You create one (once, stored in your app
//! state), call [`Overlay::render`] each frame with a closure that builds
//! the panel interior, and drive it with [`Overlay::open`] /
//! [`Overlay::close`] / [`Overlay::toggle`] from button handlers. The
//! tween key and opacity signal are private — you never see them.
//!
//! ## Multiple overlays / stacking
//! Each `Overlay` is independent (its own opacity signal + auto-allocated
//! tween key), so any number can be open at once. They **stack by render
//! order**: render them in the order you want them layered (last = on
//! top). Each one's full-window scrim blocks input to everything rendered
//! before it, so the topmost overlay owns input until dismissed, then the
//! one below takes over — standard modal-stack behaviour, no extra
//! bookkeeping. (Relies on topmost-wins hit-testing; see `input::hit_test`.)
//!
//! ## Cost
//! Closed (opacity 0) → the scrim's [`crate::scene::NodeBuilderRef::opacity_bind`]
//! marks the subtree invisible, so flatten skips it entirely: no draw, no
//! hit entries, no per-frame work. Open + idle → one full-window scrim
//! quad + the panel; the loop parks (0% CPU) until the next input. Only
//! the ~160 ms fade animates.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use winit::window::CursorIcon;

use crate::anim::{Curve, Timeline};
use crate::layout::{Align, Justify, Len};
use crate::scene::Scene;
use crate::signal::Signal;

/// Base of the tween-key range handed out to engine widgets. Kept far
/// above the small hand-picked keys apps use for their own tweens (those
/// are conventionally in the low range, e.g. `0x0001_xxxx`) so the two
/// can never collide.
const WIDGET_TWEEN_BASE: u32 = 0xF000_0000;
static NEXT_WIDGET_KEY: AtomicU32 = AtomicU32::new(WIDGET_TWEEN_BASE);

/// A modal overlay. Cheap to clone (an `Rc`-backed signal + a few `Copy`
/// fields) — clone it into button handlers freely.
#[derive(Clone)]
pub struct Overlay {
    /// Group opacity 0..=1, tweened on open/close and bound to the scrim
    /// root (cascades to the whole subtree). Private: the only way to move
    /// it is open/close/toggle/reset.
    opacity: Signal<f32>,
    /// Stable, process-unique timeline key for this overlay's fade. Auto-
    /// allocated in [`Overlay::new`] so callers never pick magic numbers
    /// or risk collisions.
    tween_key: u32,
    fade: Duration,
    curve: Curve,
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    /// Create a closed overlay with a 160 ms ease-in-out fade.
    pub fn new() -> Self {
        Self {
            opacity: Signal::new(0.0),
            tween_key: NEXT_WIDGET_KEY.fetch_add(1, Ordering::Relaxed),
            fade: Duration::from_millis(160),
            curve: Curve::EaseInOut,
        }
    }

    /// Override the fade duration (applies to both directions).
    pub fn with_fade(mut self, fade: Duration) -> Self {
        self.fade = fade;
        self
    }

    /// Override the fade easing curve.
    pub fn with_curve(mut self, curve: Curve) -> Self {
        self.curve = curve;
        self
    }

    /// Whether the overlay is logically open (more than half faded in).
    /// Used by [`Self::toggle`] and for "is a modal up?" checks.
    pub fn is_open(&self) -> bool {
        self.opacity.get() > 0.5
    }

    /// Fade in. Idempotent + interruptible: re-targets the same tween, so
    /// opening mid-close smoothly reverses from wherever it is.
    pub fn open(&self, timeline: &mut Timeline, now: Instant) {
        timeline.start(self.tween_key, self.opacity.clone(), 1.0, self.curve, self.fade, now);
    }

    /// Fade out. Same interruptible semantics as [`Self::open`].
    pub fn close(&self, timeline: &mut Timeline, now: Instant) {
        timeline.start(self.tween_key, self.opacity.clone(), 0.0, self.curve, self.fade, now);
    }

    /// Open if closed, close if open.
    pub fn toggle(&self, timeline: &mut Timeline, now: Instant) {
        if self.is_open() {
            self.close(timeline, now);
        } else {
            self.open(timeline, now);
        }
    }

    /// Snap shut with no animation. For when the surrounding context is
    /// being torn down anyway (e.g. signing out / changing views) and a
    /// fade would be pointless or run against an unmounted tree.
    pub fn reset(&self) {
        self.opacity.set(0.0);
    }

    /// Snap fully open with no animation — e.g. restoring a modal that
    /// was open when the app last closed, where a fade-in would look like
    /// a glitch on launch.
    pub fn open_instant(&self) {
        self.opacity.set(1.0);
    }

    /// Render the modal. Builds a full-window scrim (dimmed by
    /// `scrim_color`, click-to-dismiss, input-blocking) with the panel
    /// `content` centred on top; the whole thing fades via the owned
    /// opacity. `content` builds the panel interior — style it however you
    /// like; the overlay handles the scrim, dismissal, centring and the
    /// click-absorbing host so panel clicks never leak to the dismiss
    /// handler.
    ///
    /// **Call this last** in your scene (after all normal content) so it
    /// layers on top. For stacked modals, render them back-to-front.
    /// Always safe to call every frame: when closed it's skipped by the
    /// opacity-visibility gate.
    pub fn render<F: FnOnce(&mut Scene)>(
        &self,
        s: &mut Scene,
        scrim_color: [f32; 4],
        content: F,
    ) {
        let opacity = self.opacity.clone();
        let key = self.tween_key;
        let fade = self.fade;
        let curve = self.curve;
        s.col(())
            .abs(0.0, 0.0)
            .w(Len::Fill)
            .h(Len::Fill)
            .rgba(scrim_color[0], scrim_color[1], scrim_color[2], scrim_color[3])
            // Fades the whole subtree; at ~0 it also drops the scrim from
            // flatten, so a closed modal neither draws nor eats input.
            .opacity_bind(opacity.clone())
            .align(Align::Center)
            .justify(Justify::Center)
            // Click outside the panel dismisses. (Topmost-wins hit-testing
            // means this only fires when the cursor isn't over the panel.)
            .on_click(move |ctx| {
                ctx.timeline
                    .start(key, opacity.clone(), 0.0, curve, fade, ctx.now);
            })
            // The dim area is click-to-dismiss but shouldn't read as a
            // button — keep the normal arrow over it (overrides the
            // auto-pointer the `on_click` would otherwise trigger).
            .cursor(CursorIcon::Default)
            .child(|scrim| {
                // Absorbing host: a click anywhere on the panel (including
                // its empty padding, which isn't itself a hit target)
                // lands here instead of falling through to the scrim's
                // dismiss handler. Same arrow-cursor reasoning as the
                // scrim — only the real buttons inside should show pointer.
                scrim
                    .col(())
                    .on_click(|_| {})
                    .cursor(CursorIcon::Default)
                    .child(content);
            });
    }
}
