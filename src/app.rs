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
use crate::gpu::{FrameStats, GpuContext, ShapeInstance};
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
    /// Last dirty bitmask consumed by `flush_tree`. Cleared on read by
    /// `take_dirty`, so we cache it for later stat dumps.
    last_dirty_mask: u32,
    /// Wall-clock CPU time of the most recent `render_once` call.
    last_cpu_ms: f32,
    /// Stat-dump cadence: continuously log on every render when set.
    /// Toggled by `FROSTIFY_STATS=1` env var or by F1 in interactive mode.
    stats_log: bool,
    /// Bar-gauge HUD overlay: enabled by F1 (along with stats logging).
    /// Stage-1 has no text renderer, so the HUD is rect-only.
    hud_enabled: bool,
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
            last_dirty_mask: 0,
            last_cpu_ms: 0.0,
            stats_log: std::env::var_os("FROSTIFY_STATS").is_some(),
            hud_enabled: std::env::var_os("FROSTIFY_HUD").is_some(),
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
        let mask = self.ctx.tree.take_dirty();
        if mask == 0 {
            return false;
        }
        self.last_dirty_mask = mask;
        let (flat, opaque_count, hits) = self.ctx.tree.flatten();
        self.instances = flat;
        self.opaque_count = opaque_count;
        self.hits = hits;
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_instances(&self.instances, self.opaque_count, backdrop_hint);
        }
        true
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn render_once(&mut self) {
        if self.hud_enabled {
            self.refresh_hud_overlay();
        }
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let t0 = Instant::now();
        gpu.render_frame();
        self.last_cpu_ms = t0.elapsed().as_secs_f32() * 1_000.0;
        if self.stats_log {
            log::info!("frame stats: {:?}", self.current_stats());
        }
    }

    /// Build a small set of debug rectangles representing the most
    /// recent frame stats and upload them as overlay instances. Cleared
    /// when `hud_enabled` flips off.
    fn refresh_hud_overlay(&mut self) {
        let stats = self.current_stats();
        let instances = build_hud_instances(&stats);
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_overlay_instances(&instances);
        }
    }

    fn clear_hud_overlay(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_overlay_instances(&[]);
        }
    }

    /// Forward the GPU memory-allocation snapshot from the renderer.
    /// Returns `None` if the GPU isn't initialised yet.
    pub fn memory_report(&self) -> Option<crate::gpu::MemoryReport> {
        self.gpu.as_ref().map(|g| g.memory_report())
    }

    /// Combine the renderer's GPU stats with the app-side CPU + dirty mask.
    pub fn current_stats(&self) -> FrameStats {
        let mut s = self
            .gpu
            .as_ref()
            .map(|g| g.last_frame_stats())
            .unwrap_or_default();
        s.cpu_ms = self.last_cpu_ms;
        s.dirty_mask = self.last_dirty_mask;
        s
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
        let stats = self.current_stats();
        debug::write_stats_sidecar(&path, &stats);
    }
}

/// Bar-gauge HUD layout. Stage-1 is rect-only — see PLAN.md "no text"
/// rule. Each metric gets a horizontal bar; the fill width is a
/// fraction (0..=1) of a per-metric reference cap. Color shifts from
/// green → yellow → red as the fraction crosses 0.5 / 0.8.
fn build_hud_instances(stats: &FrameStats) -> Vec<ShapeInstance> {
    const ORIGIN_X: f32 = 12.0;
    const ORIGIN_Y: f32 = 12.0;
    const BAR_W: f32 = 200.0;
    const BAR_H: f32 = 10.0;
    const BAR_GAP: f32 = 6.0;
    const PAD: f32 = 8.0;

    let metrics = [
        ("cpu", stats.cpu_ms / 16.6),
        ("gpu", stats.gpu_ms / 16.6),
        ("inst", stats.instance_count as f32 / 256.0),
        ("draw", stats.drawcalls as f32 / 16.0),
    ];

    let panel_w = BAR_W + PAD * 2.0;
    let panel_h = (BAR_H + BAR_GAP) * metrics.len() as f32 + PAD * 2.0 - BAR_GAP;

    let mut out = Vec::with_capacity(1 + metrics.len() * 2);

    // Background panel.
    out.push(ShapeInstance {
        color: [0.0, 0.0, 0.0, 0.6],
        position: [ORIGIN_X, ORIGIN_Y],
        size: [panel_w, panel_h],
        border_radius: [6.0; 4],
        ..Default::default()
    });

    for (i, (_, frac)) in metrics.iter().enumerate() {
        let bar_y = ORIGIN_Y + PAD + i as f32 * (BAR_H + BAR_GAP);
        let bar_x = ORIGIN_X + PAD;
        // Bar background.
        out.push(ShapeInstance {
            color: [1.0, 1.0, 1.0, 0.10],
            position: [bar_x, bar_y],
            size: [BAR_W, BAR_H],
            border_radius: [3.0; 4],
            ..Default::default()
        });
        // Bar fill.
        let f = frac.clamp(0.0, 1.0);
        let fill_w = (BAR_W * f).max(0.0);
        let color = if f < 0.5 {
            [0.30, 0.85, 0.40, 1.0]
        } else if f < 0.8 {
            [0.95, 0.80, 0.20, 1.0]
        } else {
            [0.95, 0.30, 0.30, 1.0]
        };
        if fill_w > 0.0 {
            out.push(ShapeInstance {
                color,
                position: [bar_x, bar_y],
                size: [fill_w, BAR_H],
                border_radius: [3.0; 4],
                ..Default::default()
            });
        }
    }

    out
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
        let mask = self.ctx.tree.take_dirty();
        if mask == 0 {
            return false;
        }
        let (flat, opaque_count, hits) = self.ctx.tree.flatten();
        *self.instances = flat;
        *self.opaque_count = opaque_count;
        *self.hits = hits;
        let backdrop_hint = mask & crate::node::dirty::BACKDROP != 0;
        self.gpu
            .set_instances(self.instances, *self.opaque_count, backdrop_hint);
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
        let mut stats = self.gpu.last_frame_stats();
        // CPU ms isn't tracked here — capture path is non-hot. Leave at 0.0.
        stats.dirty_mask = 0;
        debug::write_stats_sidecar(&path, &stats);
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

    /// Build + upload the bar-gauge HUD overlay from the renderer's
    /// most recent frame stats. Calls `set_overlay_instances`.
    pub fn show_hud(&mut self) {
        let stats = self.gpu.last_frame_stats();
        let inst = build_hud_instances(&stats);
        self.gpu.set_overlay_instances(&inst);
    }

    /// Clear any active HUD overlay.
    pub fn hide_hud(&mut self) {
        self.gpu.set_overlay_instances(&[]);
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

        if let Some(mem) = self.memory_report() {
            log::info!(
                "gpu memory: total={} KiB (instance={} overlay={} blur={} overdraw={} timing={} prev_cpu={})",
                mem.total() / 1024,
                mem.instance_buffer,
                mem.overlay_buffer,
                mem.blur_textures,
                mem.overdraw_textures,
                mem.timing,
                mem.prev_instances_cpu,
            );
        }

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
                        KeyCode::F1 => {
                            self.hud_enabled = !self.hud_enabled;
                            self.stats_log = self.hud_enabled;
                            log::info!(
                                "hud/stats: {} | last frame: {:?}",
                                if self.hud_enabled { "on" } else { "off" },
                                self.current_stats()
                            );
                            if !self.hud_enabled {
                                self.clear_hud_overlay();
                            }
                            self.request_redraw();
                            return;
                        }
                        KeyCode::F2 => {
                            self.save_screenshot();
                            return;
                        }
                        KeyCode::F4 => {
                            if let Some(g) = self.gpu.as_mut() {
                                let on = !g.overdraw_mode();
                                g.set_overdraw(on);
                                log::info!("overdraw heatmap: {}", if on { "on" } else { "off" });
                            }
                            self.request_redraw();
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
