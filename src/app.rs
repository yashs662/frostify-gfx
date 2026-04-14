//! Application shell — wraps winit + wgpu + scene + binds + input + anim.
//!
//! Use it like:
//!
//! ```ignore
//! frostify_gfx::app::App::new("demo", 1100, 750)
//!     .scene(|s| build_demo(s))
//!     .on_key(|code, state, ctx| handle_key(code, state, ctx))
//!     .run()
//!     .unwrap();
//! ```
//!
//! The app owns:
//! - a `SceneCtx` (tree + name index + bind registry) populated by the
//!   user-supplied `scene` closure;
//! - a `winit` window + `GpuContext` lazily created in `resumed`;
//! - an `InputState` driven by the window's pointer events;
//! - a `Timeline` whose tweens target the per-bind "displayed" signals
//!   that the scene allocated when an `animated(...)` color was set.
//!
//! The shell wires these together through a small set of helpers — the
//! one place to look when changing the reactive flow is
//! [`App::process_binds`] (snap or retarget) plus
//! [`App::pump_animated_displays`] (push interpolated values into the
//! tree on each anim tick). Everything else is winit boilerplate.
//!
//! Built-in keys: `Esc` exits, `F2` writes a screenshot under the
//! configured `capture_dir`, `F5` forces a full tree rebuild + redraw.
//! Pass `.on_key(...)` for additional bindings; pass `.headless(...)`
//! for scripted offscreen captures (used by CI/self-verify).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::anim::Timeline;
use crate::debug;
use crate::gpu::{GpuContext, ShapeInstance};
use crate::input::InputState;
use crate::node::HitEntry;
use crate::scene::{ColorBindSlot, Scene, SceneCtx};

/// Tween-key namespace reserved for the bind registry. User-chosen
/// keys should stay below this. One key per bind slot index.
const BIND_TWEEN_KEY_BASE: u32 = 0xC000_0000;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub decorations: bool,
    pub transparent: bool,
    pub blur: bool,
    pub capture_dir: PathBuf,
}

impl AppConfig {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            decorations: false,
            transparent: true,
            blur: true,
            capture_dir: PathBuf::from("debug_captures"),
        }
    }
}

/// Optional headless script. Called once from `resumed` after the
/// window is ready and the first frame has been captured. Use it to
/// drive synthetic input, mutate the tree, advance the timeline by
/// hand, and capture additional frames. After the closure returns,
/// the event loop exits.
pub type HeadlessFn = Box<dyn FnOnce(&mut HeadlessHelper)>;

/// Optional per-keypress hook. Called on `Pressed` events for any
/// `KeyCode` the shell hasn't already handled (Esc/F2/F5 are
/// reserved). After the closure returns, the shell processes binds,
/// flushes the tree, and requests a redraw if anything changed.
pub type KeyFn = Box<dyn FnMut(KeyCode, ElementState, &mut SceneCtx)>;

pub struct App {
    config: AppConfig,
    ctx: SceneCtx,
    instances: Vec<ShapeInstance>,
    opaque_count: u32,
    hits: Vec<HitEntry>,
    input: InputState,
    timeline: Timeline,
    on_key: Option<KeyFn>,
    headless: Option<HeadlessFn>,
    /// Number of still frames to capture and exit after, or `None` for
    /// a normal interactive run. `Some(n)` triggers headless capture
    /// mode in `resumed`.
    capture_frames: Option<u32>,
    // Lazy:
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
}

impl App {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            config: AppConfig::new(title, width, height),
            ctx: SceneCtx::new(),
            instances: Vec::new(),
            opaque_count: 0,
            hits: Vec::new(),
            input: InputState::new(),
            timeline: Timeline::new(),
            on_key: None,
            headless: None,
            capture_frames: None,
            window: None,
            gpu: None,
        }
    }

    /// Run the user-supplied scene builder. Mutates the inner
    /// `SceneCtx` immediately — by the time `run` is called the tree
    /// and the bind registry are fully populated.
    pub fn scene<F: FnOnce(&mut Scene)>(mut self, f: F) -> Self {
        let mut scene = Scene::root(&mut self.ctx);
        f(&mut scene);
        self
    }

    /// Provide a closure invoked on every key-down event for keys the
    /// shell hasn't already consumed. The closure can mutate signals,
    /// the tree, or the scene context directly.
    pub fn on_key<F: FnMut(KeyCode, ElementState, &mut SceneCtx) + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_key = Some(Box::new(f));
        self
    }

    /// Configure the directory used by F2 screenshots and headless
    /// captures.
    pub fn capture_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.capture_dir = dir.into();
        self
    }

    /// Capture `frames` still snapshots of the initial scene under
    /// `capture_dir` and exit. For scripted scenarios that mutate
    /// state between frames use [`App::headless`] instead. `frames`
    /// must be ≥ 1; the first frame is always written before any
    /// scripted callback runs.
    pub fn capture(mut self, frames: u32, dir: impl Into<PathBuf>) -> Self {
        self.capture_frames = Some(frames.max(1));
        self.config.capture_dir = dir.into();
        self
    }

    /// Convenience: capture one frame to the default `capture_dir`
    /// and exit. Equivalent to `capture(1, "debug_captures")`.
    pub fn capture_once(self) -> Self {
        self.capture(1, "debug_captures")
    }

    /// Env-var shim for the legacy `FROSTIFY_AUTOCAPTURE` flag.
    /// Returns `self` unchanged when the variable is not set, so the
    /// call is harmless in normal interactive runs. CI/self-verify
    /// keeps working without code changes; scripted multi-frame flows
    /// still use [`App::headless`] separately.
    pub fn capture_from_env(self) -> Self {
        if std::env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
            self.capture_once()
        } else {
            self
        }
    }

    /// Provide a one-shot scripted headless callback. The closure
    /// receives a `HeadlessHelper` with mutable access to every
    /// piece of shell state, runs whatever sequence it likes, and
    /// returns. The shell then exits the event loop. Implies
    /// `capture_once()` so the initial frame is always saved before
    /// the script runs.
    pub fn headless<F: FnOnce(&mut HeadlessHelper) + 'static>(mut self, f: F) -> Self {
        self.headless = Some(Box::new(f));
        if self.capture_frames.is_none() {
            self.capture_frames = Some(1);
        }
        self
    }

    /// Window decoration flag. Default `false` (frameless).
    pub fn decorations(mut self, on: bool) -> Self {
        self.config.decorations = on;
        self
    }

    /// Get a read-only view of the scene context — useful for tests
    /// that build the scene then assert on it without ever calling
    /// `run`.
    pub fn ctx(&self) -> &SceneCtx {
        &self.ctx
    }

    /// Take the scene context out of the app. Intended as the
    /// escape hatch when a consumer wants to drive winit + wgpu
    /// themselves but still wants the declarative scene builder.
    pub fn into_ctx(self) -> SceneCtx {
        self.ctx
    }

    /// Run the event loop. Blocks until the window closes or the
    /// headless script exits.
    pub fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = EventLoop::new()?;
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    // ---- Internal helpers ----------------------------------------------

    /// Re-flatten + upload the tree if any dirty flag is set.
    fn flush_tree(&mut self) -> bool {
        if self.ctx.tree.take_dirty() == 0 {
            return false;
        }
        let (flat, opaque_count, hits) = self.ctx.tree.flatten();
        self.instances = flat;
        self.opaque_count = opaque_count;
        self.hits = hits;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_instances(&self.instances, self.opaque_count);
        }
        true
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn render_once(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.render_frame();
        }
    }

    /// Walk every reactive bind. For each slot whose source version
    /// has advanced since last seen: snap or retarget.
    fn process_binds(&mut self, now: Instant) {
        process_color_binds(
            &mut self.ctx.binds.color,
            &mut self.ctx.tree,
            &mut self.timeline,
            now,
        );
    }

    /// For animated color binds, copy the current `displayed` signal
    /// value (driven by the timeline) into the tree. Called after
    /// every timeline tick.
    fn pump_animated_displays(&mut self) {
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
    }

    /// Common reaction sequence after any input or key event:
    /// re-target binds, push displayed values, flush, redraw.
    fn react(&mut self) {
        self.process_binds(Instant::now());
        self.pump_animated_displays();
        if self.flush_tree() {
            self.request_redraw();
        }
    }

    fn save_screenshot(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let (rgba, w, h) = gpu.capture_rgba();
        let path = debug::screenshot_path(&self.config.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("screenshot saved: {}", path.display()),
            Err(e) => log::error!("screenshot failed: {e}"),
        }
    }
}

/// Walk the color bind list. For each slot whose underlying source
/// has bumped its version: read the new target, advance
/// `last_version`, and either snap (`tree.set_color`) or start a
/// tween on the slot's `displayed` signal.
fn process_color_binds(
    slots: &mut [ColorBindSlot],
    tree: &mut crate::node::NodeTree,
    timeline: &mut Timeline,
    now: Instant,
) {
    for (idx, slot) in slots.iter_mut().enumerate() {
        let v = slot.bind.version();
        if v == slot.last_version {
            continue;
        }
        slot.last_version = v;
        let target = slot.bind.read();
        if let (Some(disp), Some((curve, dur))) = (slot.displayed.as_ref(), slot.bind.animation())
        {
            let key = BIND_TWEEN_KEY_BASE + idx as u32;
            timeline.start(key, disp.clone(), target, curve, dur, now);
        } else {
            tree.set_color(slot.node_id, target);
        }
    }
}

/// Helper passed to a headless script. Bundles every piece of shell
/// state the script might need so it can drive synthetic events,
/// mutate the tree, advance the timeline, and capture frames without
/// reaching back into private fields.
pub struct HeadlessHelper<'a> {
    pub ctx: &'a mut SceneCtx,
    pub gpu: &'a mut GpuContext,
    pub input: &'a mut InputState,
    pub timeline: &'a mut Timeline,
    pub instances: &'a mut Vec<ShapeInstance>,
    pub opaque_count: &'a mut u32,
    pub hits: &'a mut Vec<HitEntry>,
    pub capture_dir: &'a Path,
}

impl<'a> HeadlessHelper<'a> {
    /// Re-flatten + upload if the tree is dirty. Returns true if it
    /// actually uploaded (matches `App::flush_tree`).
    pub fn flush(&mut self) -> bool {
        if self.ctx.tree.take_dirty() == 0 {
            return false;
        }
        let (flat, opaque_count, hits) = self.ctx.tree.flatten();
        *self.instances = flat;
        *self.opaque_count = opaque_count;
        *self.hits = hits;
        self.gpu.set_instances(self.instances, *self.opaque_count);
        true
    }

    pub fn render(&mut self) {
        self.gpu.render_frame();
    }

    pub fn capture(&mut self) {
        let (rgba, w, h) = self.gpu.capture_rgba();
        let path = debug::screenshot_path(self.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("auto-capture saved: {}", path.display()),
            Err(e) => log::error!("auto-capture failed: {e}"),
        }
    }

    /// Fast-forward every active tween to its target, push the
    /// settled values through the bind registry, and clear the
    /// timeline. Useful after a scripted state change so the next
    /// capture reflects the destination color rather than the
    /// pre-tween value.
    pub fn settle(&mut self) {
        if !self.timeline.active() {
            return;
        }
        // One tick well past any plausible duration snaps every
        // tween's signal to its `to` target.
        let _ = self
            .timeline
            .tick(Instant::now() + std::time::Duration::from_secs(10));
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
    }

    /// Run reactive bind processing + animated display pump on the
    /// shared registry. Mirrors `App::react` for scripted captures.
    pub fn react(&mut self, now: Instant) {
        process_color_binds(
            &mut self.ctx.binds.color,
            &mut self.ctx.tree,
            self.timeline,
            now,
        );
        for slot in &self.ctx.binds.color {
            if let Some(disp) = &slot.displayed {
                self.ctx.tree.set_color(slot.node_id, disp.get());
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Wait);

        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_transparent(self.config.transparent)
            .with_decorations(self.config.decorations)
            .with_resizable(true)
            .with_blur(self.config.blur)
            .with_visible(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_arc: Arc<Window> = Arc::new(window);
        self.gpu = Some(GpuContext::new(Arc::clone(&window_arc)));
        self.window = Some(window_arc);

        // First flush + render so the visible window already shows
        // the scene by the time the user sees it.
        self.flush_tree();
        self.render_once();
        if let Some(w) = &self.window {
            w.set_visible(true);
            w.request_redraw();
        }

        if let Some(n) = self.capture_frames {
            // First frame is always written before the script (if any).
            self.save_screenshot();
            // Additional N-1 still frames for plain `.capture(N, ...)`
            // mode. Skipped when a headless script is also attached
            // because the script is responsible for its own captures.
            if self.headless.is_none() {
                for _ in 1..n {
                    self.render_once();
                    self.save_screenshot();
                }
            }
        }

        if let Some(script) = self.headless.take() {
            let mut helper = HeadlessHelper {
                ctx: &mut self.ctx,
                gpu: self.gpu.as_mut().expect("gpu"),
                input: &mut self.input,
                timeline: &mut self.timeline,
                instances: &mut self.instances,
                opaque_count: &mut self.opaque_count,
                hits: &mut self.hits,
                capture_dir: &self.config.capture_dir,
            };
            script(&mut helper);
            event_loop.exit();
            return;
        }

        if self.capture_frames.is_some() {
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(_gpu) = self.gpu.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(size.width, size.height);
                }
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.render_once();
            }
            WindowEvent::Occluded(occluded) => {
                if !occluded {
                    self.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                let change =
                    self.input
                        .on_cursor_moved(x, y, &self.hits, &self.ctx.tree);
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                let change = self.input.on_cursor_left(&self.hits, &self.ctx.tree);
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let change = match state {
                    ElementState::Pressed => {
                        self.input.on_left_pressed(&self.hits, &self.ctx.tree)
                    }
                    ElementState::Released => {
                        self.input.on_left_released(&self.hits, &self.ctx.tree)
                    }
                };
                if change.any() {
                    self.react();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                if state == ElementState::Pressed {
                    match code {
                        KeyCode::Escape => {
                            event_loop.exit();
                            return;
                        }
                        KeyCode::F2 => {
                            self.save_screenshot();
                            return;
                        }
                        KeyCode::F5 => {
                            self.ctx.tree.mark_all_dirty();
                            if self.flush_tree() {
                                self.request_redraw();
                            }
                            return;
                        }
                        _ => {}
                    }
                }
                if let Some(handler) = self.on_key.as_mut() {
                    handler(code, state, &mut self.ctx);
                    self.react();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Animation pump. If idle, park on `Wait` so the loop is 0% CPU.
        // If active, advance every tween, push interpolated values
        // through the bind registry, flush, and request redraw with
        // the next frame deadline.
        if !self.timeline.active() {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let res = self.timeline.tick(Instant::now());
        if res.updated {
            self.pump_animated_displays();
            if self.flush_tree() {
                self.request_redraw();
            }
        }
        match res.next_deadline {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}
