//! frostify-gfx demo: transparent wgpu window, retained tree, signals,
//! glass, and pointer-input reactivity.
//!
//! All demo-specific code (scene building, hero reactions, autocapture
//! env vars) lives here. The library itself exports only primitives
//! (`GpuContext`, `NodeTree`, `Signal`, `InputState`, …). This file is a
//! complete reference integration you can copy into a real app.
//!
//! Controls:
//!   Mouse           Hover / click the hero rect to recolor it.
//!   Space           Toggle the hero "lit" color.
//!   Arrow Left/Right Move the cyan blob behind the frosted glass.
//!   F2              Save a PNG screenshot to `debug_captures/`.
//!   F5              Force a full tree rebuild + redraw.
//!   Esc             Exit.
//!
//! Headless verification env vars (each adds one or more frames to
//! `debug_captures/`):
//!   FROSTIFY_AUTOCAPTURE=1         → render once + capture, then exit.
//!   FROSTIFY_AUTOCAPTURE_HIT=1     → also synthesize hover + press.
//!   FROSTIFY_AUTOCAPTURE_GLASS=1   → also move the blob + capture.
//!   FROSTIFY_AUTOCAPTURE_TOGGLE=1  → also toggle hero_lit + capture.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use frostify_gfx::{
    debug, Curve, GpuContext, HitEntry, InputState, Node, NodeId, NodeTree, ShapeInstance, Signal,
    Timeline,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const HOVER_TWEEN_KEY: u32 = 1;
const PRESS_TWEEN_KEY: u32 = 2;
const HOVER_TWEEN_MS: u64 = 220;

type WindowArc = Arc<Window>;

#[derive(Clone, Debug)]
struct DemoConfig {
    title: String,
    width: u32,
    height: u32,
    decorations: bool,
    capture_dir: PathBuf,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            title: "frostify-gfx".into(),
            width: 1100,
            height: 750,
            decorations: false,
            capture_dir: PathBuf::from("debug_captures"),
        }
    }
}

struct DemoIds {
    hero: NodeId,
    /// Colored rect behind the frosted glass panel. Left/Right arrow
    /// drives its X position; its motion is what the blur reveals.
    bg_blob: NodeId,
}

struct DemoApp {
    config: DemoConfig,
    window: Option<WindowArc>,
    gpu: Option<GpuContext>,
    tree: NodeTree,
    instances: Vec<ShapeInstance>,
    opaque_count: u32,
    hits: Vec<HitEntry>,
    input: InputState,
    demo: DemoIds,
    /// Toggled by Space — alternate hero base color.
    hero_lit: Signal<bool>,
    /// Clones of the hero's hover/pressed signals so the demo can read
    /// them back in `refresh_hero_color`.
    hero_hover: Signal<bool>,
    hero_pressed: Signal<bool>,
    /// Animated 0..1 glow amount driven by the hover tween. The app
    /// reads this in `refresh_hero_color` to lerp base→bright.
    hero_glow: Signal<f32>,
    /// Animated 0..1 press depth driven by a spring on mouse press.
    hero_press_depth: Signal<f32>,
    timeline: Timeline,
    /// Scratch state for the bg-blob horizontal position.
    bg_blob_x: f32,
}

impl DemoApp {
    fn new(config: DemoConfig) -> Self {
        let hero_hover = Signal::new(false);
        let hero_pressed = Signal::new(false);
        let hero_focused = Signal::new(false);
        let (tree, demo, bg_blob_x) = build_demo_tree(
            config.width as f32,
            config.height as f32,
            hero_hover.clone(),
            hero_pressed.clone(),
            hero_focused,
        );
        log::info!("demo scene: {} nodes", tree.len());
        Self {
            config,
            window: None,
            gpu: None,
            tree,
            instances: Vec::new(),
            opaque_count: 0,
            hits: Vec::new(),
            input: InputState::new(),
            demo,
            hero_lit: Signal::new(false),
            hero_hover,
            hero_pressed,
            hero_glow: Signal::new(0.0),
            hero_press_depth: Signal::new(0.0),
            timeline: Timeline::new(),
            bg_blob_x,
        }
    }

    /// If the tree's dirty mask is set, re-flatten and push the new
    /// instance list to the GPU. Returns true if anything was uploaded.
    fn flush_tree(&mut self) -> bool {
        if self.tree.take_dirty() == 0 {
            return false;
        }
        let (flat, opaque_count, hits) = self.tree.flatten();
        self.instances = flat;
        self.opaque_count = opaque_count;
        self.hits = hits;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_instances(&self.instances, self.opaque_count);
        }
        true
    }

    /// Derive the hero color from the animated glow/press signals plus
    /// the `hero_lit` toggle, then push it through the tracked setter.
    /// Called on any input event OR any tween tick — the timeline
    /// drives this indirectly through `about_to_wait`.
    fn refresh_hero_color(&mut self) {
        let base = if self.hero_lit.get() {
            [0.20, 0.95, 0.55, 1.0]
        } else {
            [0.95, 0.25, 0.55, 1.0]
        };
        let glow = self.hero_glow.get();
        let bright = [
            (base[0] + 0.18_f32).min(1.0),
            (base[1] + 0.18_f32).min(1.0),
            (base[2] + 0.18_f32).min(1.0),
            base[3],
        ];
        let lerped = [
            base[0] * (1.0 - glow) + bright[0] * glow,
            base[1] * (1.0 - glow) + bright[1] * glow,
            base[2] * (1.0 - glow) + bright[2] * glow,
            base[3],
        ];
        // Press depth squashes brightness toward zero.
        let press = self.hero_press_depth.get();
        let press_factor = 1.0 - 0.45 * press;
        let color = [
            lerped[0] * press_factor,
            lerped[1] * press_factor,
            lerped[2] * press_factor,
            lerped[3],
        ];
        self.tree.set_color(self.demo.hero, color);
    }

    /// Call after any input event that might have flipped `hero_hover`
    /// or `hero_pressed`. Starts/retargets the corresponding tween so
    /// the derived glow/depth catches up smoothly.
    fn retarget_tweens(&mut self) {
        let now = Instant::now();
        let hover_target = if self.hero_hover.get() { 1.0 } else { 0.0 };
        if (self.hero_glow.get() - hover_target).abs() > 1e-4 {
            self.timeline.start(
                HOVER_TWEEN_KEY,
                self.hero_glow.clone(),
                hover_target,
                Curve::EaseInOut,
                Duration::from_millis(HOVER_TWEEN_MS),
                now,
            );
        }
        let press_target = if self.hero_pressed.get() { 1.0 } else { 0.0 };
        if (self.hero_press_depth.get() - press_target).abs() > 1e-4 {
            self.timeline.start(
                PRESS_TWEEN_KEY,
                self.hero_press_depth.clone(),
                press_target,
                Curve::Spring {
                    stiffness: 220.0,
                    damping: 20.0,
                },
                Duration::from_millis(600),
                now,
            );
        }
    }

    fn autocapture(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let (rgba, w, h) = gpu.capture_rgba();
        let path = debug::screenshot_path(&self.config.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("auto-capture saved: {}", path.display()),
            Err(e) => log::error!("auto-capture failed: {e}"),
        }
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

    fn run_autocapture_sequences(&mut self) {
        self.autocapture();

        if std::env::var_os("FROSTIFY_AUTOCAPTURE_HIT").is_some() {
            // Synthesize a hover + press on the hero rect to prove the
            // input plumbing drives the recolor path end-to-end. No real
            // cursor events fire in headless mode.
            let (cx, cy) = {
                let n = self.tree.get(self.demo.hero).expect("hero");
                let panel_pad = 24.0;
                (
                    panel_pad + n.position[0] + n.size[0] * 0.5,
                    panel_pad + n.position[1] + n.size[1] * 0.5,
                )
            };
            let change = self.input.on_cursor_moved(cx, cy, &self.hits, &self.tree);
            log::info!(
                "hit test at ({cx:.0},{cy:.0}) → hovered={:?}, hit cache len={}, change={:?}",
                self.input.hovered,
                self.hits.len(),
                change,
            );
            self.refresh_hero_color();
            self.flush_tree();
            self.render_once();
            self.autocapture();

            let change = self.input.on_left_pressed(&self.hits, &self.tree);
            log::info!("press change={:?}", change);
            self.refresh_hero_color();
            self.flush_tree();
            self.render_once();
            self.autocapture();

            log::info!(
                "signals: hover={} pressed={} hover.ver={} pressed.ver={}",
                self.hero_hover.get(),
                self.hero_pressed.get(),
                self.hero_hover.version(),
                self.hero_pressed.version(),
            );
        }

        if std::env::var_os("FROSTIFY_AUTOCAPTURE_GLASS").is_some() {
            // Move the bg blob horizontally and re-capture to prove
            // re-blur + glass refresh on backdrop change.
            let new_pos = {
                let Some(n) = self.tree.get(self.demo.bg_blob) else {
                    return;
                };
                [self.bg_blob_x + 120.0, n.position[1]]
            };
            self.bg_blob_x += 120.0;
            self.tree.set_position(self.demo.bg_blob, new_pos);
            self.flush_tree();
            self.render_once();
            self.autocapture();
        }

        if std::env::var_os("FROSTIFY_AUTOCAPTURE_ANIM").is_some() {
            // Drive a hover tween under manufactured time. Snapshots
            // every 40 ms for 240 ms so the ease-in-out shape is
            // visible across frames. Headless mode: no real winit
            // events ever fire, so we call tick() directly.
            self.hero_hover.set(true);
            self.retarget_tweens();
            let t0 = Instant::now();
            for step in 1..=6u32 {
                let sim = t0 + Duration::from_millis(step as u64 * 40);
                let res = self.timeline.tick(sim);
                log::info!(
                    "anim step {step}: glow={:.3} updated={} active={}",
                    self.hero_glow.get(),
                    res.updated,
                    self.timeline.active(),
                );
                self.refresh_hero_color();
                self.flush_tree();
                self.render_once();
                self.autocapture();
            }
        }

        if std::env::var_os("FROSTIFY_AUTOCAPTURE_TOGGLE").is_some() {
            self.hero_lit.set(true);
            self.tree
                .set_color(self.demo.hero, [0.20, 0.95, 0.55, 1.0]);
            let mask_before = self.tree.dirty();
            let flushed = self.flush_tree();
            log::info!(
                "toggle: hero_lit.version={} dirty_mask=0x{:x} flushed={}",
                self.hero_lit.version(),
                mask_before,
                flushed,
            );
            self.render_once();
            self.autocapture();
        }
    }
}

/// Demo scene: dark panel + title bar + 8×5 swatch grid + hero accent
/// rect + cyan blob behind a frosted glass rect.
fn build_demo_tree(
    w: f32,
    h: f32,
    hero_hover: Signal<bool>,
    hero_pressed: Signal<bool>,
    hero_focused: Signal<bool>,
) -> (NodeTree, DemoIds, f32) {
    let mut tree = NodeTree::new();

    let panel_pad = 24.0;
    let panel = tree.add_root(
        Node::rect()
            .pos(panel_pad, panel_pad)
            .size(w - panel_pad * 2.0, h - panel_pad * 2.0)
            .rgba(0.00, 0.00, 0.00, 0.5)
            .radius(28.0)
            .border(1.5, [1.0, 1.0, 1.0, 0.10])
            .shadow([0.0, 16.0], 40.0, [0.0, 0.0, 0.0, 1.0], 0.55)
            .build(),
    );

    // Title bar.
    tree.add_child(
        panel,
        Node::rect()
            .pos(20.0, 20.0)
            .size(w - panel_pad * 2.0 - 40.0, 56.0)
            .rgba(0.13, 0.14, 0.18, 1.0)
            .radius(16.0)
            .border(1.0, [1.0, 1.0, 1.0, 0.06])
            .build(),
    );

    // Three traffic-light style dots inside the title bar.
    let dot_y = 40.0;
    let dot_colors = [
        [0.95, 0.30, 0.30, 1.0],
        [0.95, 0.75, 0.20, 1.0],
        [0.30, 0.85, 0.40, 1.0],
    ];
    for (i, c) in dot_colors.iter().enumerate() {
        tree.add_child(
            panel,
            Node::rect()
                .pos(40.0 + i as f32 * 22.0, dot_y)
                .size(14.0, 14.0)
                .color(*c)
                .radius(7.0)
                .build(),
        );
    }

    // Swatch grid: 8 cols × 5 rows = 40 cards inside an inner row container.
    let grid_origin_x = 32.0;
    let grid_origin_y = 110.0;
    let cols = 8usize;
    let rows = 5usize;
    let cell_w = 110.0;
    let cell_h = 80.0;
    let gap = 12.0;
    let row = tree.add_child(
        panel,
        Node::rect()
            .pos(grid_origin_x, grid_origin_y)
            .size(
                cols as f32 * cell_w + (cols as f32 - 1.0) * gap,
                rows as f32 * cell_h + (rows as f32 - 1.0) * gap,
            )
            .rgba(0.0, 0.0, 0.0, 0.0)
            .build(),
    );

    for r in 0..rows {
        for c in 0..cols {
            let t = (r * cols + c) as f32 / ((rows * cols) as f32);
            let hue = t * 6.2831;
            let color = hsv_to_rgb(hue, 0.55, 0.95);
            tree.add_child(
                row,
                Node::rect()
                    .pos(c as f32 * (cell_w + gap), r as f32 * (cell_h + gap))
                    .size(cell_w, cell_h)
                    .rgba(color[0], color[1], color[2], 1.0)
                    .radius(14.0)
                    .border(1.0, [1.0, 1.0, 1.0, 0.20])
                    .shadow([0.0, 4.0], 8.0, [0.0, 0.0, 0.0, 1.0], 0.30)
                    .build(),
            );
        }
    }

    // Hero accent rect — the one interactive node in the demo.
    let hero = tree.add_child(
        panel,
        Node::rect()
            .pos(32.0, grid_origin_y + rows as f32 * (cell_h + gap) + 18.0)
            .size(380.0, 70.0)
            .rgba(0.95, 0.25, 0.55, 1.0)
            .radius(20.0)
            .border(2.0, [1.0, 1.0, 1.0, 0.85])
            .shadow([0.0, 0.0], 22.0, [0.95, 0.25, 0.55, 1.0], 0.45)
            .on_hover(hero_hover)
            .on_press(hero_pressed)
            .on_focus(hero_focused)
            .build(),
    );

    // Bright cyan blob behind a frosted glass panel. Arrow keys move it
    // to show the blur following the backdrop.
    let blob_x = w * 0.5 - 120.0;
    let blob_y = h - 260.0;
    let bg_blob = tree.add_child(
        panel,
        Node::rect()
            .pos(blob_x - panel_pad, blob_y - panel_pad)
            .size(240.0, 180.0)
            .rgba(0.10, 0.85, 0.95, 1.0)
            .radius(40.0)
            .build(),
    );
    let glass_w = 320.0;
    let glass_h = 140.0;
    tree.add_child(
        panel,
        Node::glass()
            .pos(w * 0.5 - glass_w * 0.5 - panel_pad, h - 220.0 - panel_pad)
            .size(glass_w, glass_h)
            .radius(24.0)
            .roughness(0.6)
            .rgba(1.0, 1.0, 1.0, 0.10)
            .build(),
    );

    (tree, DemoIds { hero, bg_blob }, blob_x - panel_pad)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let h6 = (h / 1.0471975) % 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r, g, b) = match h6 as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Reactive: only redraw on actual events, idle = 0% CPU.
        event_loop.set_control_flow(ControlFlow::Wait);

        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_transparent(true)
            .with_decorations(self.config.decorations)
            .with_resizable(true)
            .with_blur(true)
            .with_visible(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));

        let window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_arc: WindowArc = Arc::new(window);

        let gpu = GpuContext::new(Arc::clone(&window_arc));
        self.gpu = Some(gpu);
        self.flush_tree();
        log::info!(
            "first upload: {} instances ({} bytes)",
            self.instances.len(),
            self.instances.len() * std::mem::size_of::<ShapeInstance>()
        );
        self.render_once();
        window_arc.set_visible(true);
        window_arc.request_redraw();
        self.window = Some(window_arc);

        if std::env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
            self.run_autocapture_sequences();
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                gpu.render_frame();
            }
            WindowEvent::Occluded(occluded) => {
                if !occluded {
                    self.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x as f32;
                let y = position.y as f32;
                let change = self.input.on_cursor_moved(x, y, &self.hits, &self.tree);
                if change.any() {
                    self.retarget_tweens();
                    self.refresh_hero_color();
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                let change = self.input.on_cursor_left(&self.hits, &self.tree);
                if change.any() {
                    self.retarget_tweens();
                    self.refresh_hero_color();
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let change = match state {
                    ElementState::Pressed => self.input.on_left_pressed(&self.hits, &self.tree),
                    ElementState::Released => self.input.on_left_released(&self.hits, &self.tree),
                };
                if change.any() {
                    self.retarget_tweens();
                    self.refresh_hero_color();
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match code {
                KeyCode::F2 => {
                    let (rgba, w, h) = gpu.capture_rgba();
                    let path = debug::screenshot_path(&self.config.capture_dir);
                    match debug::save_png(&path, &rgba, w, h) {
                        Ok(()) => log::info!("screenshot saved: {}", path.display()),
                        Err(e) => log::error!("screenshot failed: {e}"),
                    }
                }
                KeyCode::F5 => {
                    self.tree.mark_all_dirty();
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
                KeyCode::Space => {
                    let lit = !self.hero_lit.get();
                    self.hero_lit.set(lit);
                    self.refresh_hero_color();
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
                KeyCode::ArrowLeft | KeyCode::ArrowRight => {
                    let delta = if matches!(code, KeyCode::ArrowLeft) {
                        -20.0
                    } else {
                        20.0
                    };
                    self.bg_blob_x += delta;
                    let pos = {
                        let Some(n) = self.tree.get(self.demo.bg_blob) else {
                            return;
                        };
                        [self.bg_blob_x, n.position[1]]
                    };
                    self.tree.set_position(self.demo.bg_blob, pos);
                    if self.flush_tree() {
                        self.request_redraw();
                    }
                }
                KeyCode::Escape => event_loop.exit(),
                _ => {}
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // M6 animation pump. If the timeline is idle we park on
        // `Wait` — the whole point of the reactive loop is 0% CPU at
        // rest. If any tween is running we tick it, push derived
        // values into the tree, request a redraw, and hand winit the
        // next frame deadline via `WaitUntil`.
        if !self.timeline.active() {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let res = self.timeline.tick(Instant::now());
        if res.updated {
            self.refresh_hero_color();
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();
    let event_loop = EventLoop::new()?;
    let mut app = DemoApp::new(DemoConfig::default());
    event_loop.run_app(&mut app)?;
    Ok(())
}
