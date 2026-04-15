//! frostify-gfx demo — declarative scene + reactive bindings via the
//! library's `App` shell.
//!
//! All winit/gpu/flush/input/animation plumbing lives in `frostify_gfx`.
//! This file is just: signals, a scene closure, a key handler, and a
//! one-shot headless script for CI captures.
//!
//! Controls:
//!   Mouse           Hover / click the hero rect to recolor it.
//!   Space           Toggle the hero "lit" base color.
//!   Arrow Left/Right Move the cyan blob behind the frosted glass.
//!   F2              Save a PNG screenshot to `debug_captures/`.
//!   F5              Force a full tree rebuild + redraw.
//!   Esc             Exit.
//!
//! Headless verification env vars (each adds one or more frames to
//! `debug_captures/`):
//! CLI flags (override env vars when present):
//!   --capture frames=N out=DIR     → render N frames + exit.
//!
//! Headless verification env vars (each adds one or more frames to
//! `debug_captures/`):
//!   FROSTIFY_AUTOCAPTURE=1         → render once + capture, then exit.
//!   FROSTIFY_AUTOCAPTURE_HIT=1     → also synthesize hover + press.
//!   FROSTIFY_AUTOCAPTURE_GLASS=1   → also move the blob + capture.
//!   FROSTIFY_AUTOCAPTURE_TOGGLE=1  → also toggle hero_lit + capture.
//!   FROSTIFY_AUTOCAPTURE_ANIM=1    → also drive a hover tween + capture.
//!   FROSTIFY_AUTOCAPTURE_OVERDRAW=1 → also enable the F4 heatmap + capture.

use std::cell::Cell;
use std::env;
use std::rc::Rc;
use std::time::{Duration, Instant};

use frostify_gfx::{
    animated, App, Computed, Curve, HeadlessHelper, Scene, Signal,
};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const W: f32 = 1100.0;
const H: f32 = 750.0;
const PAD: f32 = 24.0;
const DOTS: [[f32; 4]; 3] = [
    [0.95, 0.30, 0.30, 1.0],
    [0.95, 0.75, 0.20, 1.0],
    [0.30, 0.85, 0.40, 1.0],
];
const BLOB_Y: f32 = 240.0;
const BLOB_X0: f32 = 330.0;

#[derive(Clone)]
struct Sigs {
    lit: Signal<bool>,
    hover: Signal<bool>,
    pressed: Signal<bool>,
    focused: Signal<bool>,
}

impl Sigs {
    fn new() -> Self {
        Self {
            lit: Signal::new(false),
            hover: Signal::new(false),
            pressed: Signal::new(false),
            focused: Signal::new(false),
        }
    }
}

/// Hero color derivation: base color (toggled by Space), brightened on
/// hover, darkened on press. The `Computed` recomputes lazily whenever
/// any of the three deps bumps its version, and the animated bind in
/// `build_scene` tweens its output smoothly.
fn hero_color(sigs: &Sigs) -> Computed<[f32; 4]> {
    Computed::new(
        (sigs.lit.clone(), sigs.hover.clone(), sigs.pressed.clone()),
        |(l, h, p)| {
            let base = if l {
                [0.20, 0.95, 0.55, 1.0]
            } else {
                [0.95, 0.25, 0.55, 1.0]
            };
            let lerped = if h {
                [
                    (base[0] + 0.18_f32).min(1.0),
                    (base[1] + 0.18_f32).min(1.0),
                    (base[2] + 0.18_f32).min(1.0),
                    base[3],
                ]
            } else {
                base
            };
            let pf = if p { 0.55_f32 } else { 1.0 };
            [lerped[0] * pf, lerped[1] * pf, lerped[2] * pf, lerped[3]]
        },
    )
}

/// Declarative scene: panel + title bar + 3 dots + interactive hero +
/// frosted glass over a moving blob. Roughly 30 lines of meaningful
/// content — everything else is plumbing the library handles.
fn build_scene(s: &mut Scene, sigs: &Sigs) {
    let hero = hero_color(sigs);
    s.rect("root")
        .pos(PAD, PAD)
        .size(W - PAD * 2.0, H - PAD * 2.0)
        .rgba(0.0, 0.0, 0.0, 0.5)
        .radius(28.0)
        .border(1.5, [1.0, 1.0, 1.0, 0.10])
        .shadow([0.0, 16.0], 40.0, [0.0, 0.0, 0.0, 1.0], 0.55)
        .child(|p| {
            p.rect("title")
                .pos(20.0, 20.0)
                .size(W - PAD * 2.0 - 40.0, 56.0)
                .rgba(0.13, 0.14, 0.18, 1.0)
                .radius(16.0)
                .border(1.0, [1.0, 1.0, 1.0, 0.06]);
            for (i, c) in DOTS.iter().enumerate() {
                p.rect("")
                    .pos(40.0 + i as f32 * 22.0, 40.0)
                    .size(14.0, 14.0)
                    .color(*c)
                    .radius(7.0);
            }
            p.rect("hero")
                .pos(32.0, 110.0)
                .size(380.0, 70.0)
                .color(animated(hero, Curve::EaseInOut, Duration::from_millis(220)))
                .radius(20.0)
                .border(2.0, [1.0, 1.0, 1.0, 0.85])
                .shadow([0.0, 0.0], 22.0, [0.95, 0.25, 0.55, 1.0], 0.45)
                .on_hover(sigs.hover.clone())
                .on_press(sigs.pressed.clone())
                .on_focus(sigs.focused.clone());
            p.rect("blob")
                .pos(BLOB_X0, BLOB_Y)
                .size(240.0, 180.0)
                .rgba(0.10, 0.85, 0.95, 1.0)
                .radius(40.0);
            p.glass("glass")
                .pos(BLOB_X0 + 36.0, BLOB_Y + 40.0)
                .size(320.0, 140.0)
                .radius(24.0)
                .roughness(0.6)
                .rgba(1.0, 1.0, 1.0, 0.10);
        });
}

/// Headless capture script — replaces the env-var-driven sequences in
/// the old example. Runs whichever sub-flows the env vars asked for,
/// then returns; the shell exits the event loop afterwards.
fn run_headless(h: &mut HeadlessHelper, sigs: Sigs, blob_x: Rc<Cell<f32>>) {
    if env::var_os("FROSTIFY_AUTOCAPTURE_HIT").is_some() {
        let (cx, cy) = match h.ctx.node("hero").and_then(|id| h.ctx.tree.get(id)) {
            Some(n) => (
                PAD + n.position[0] + n.size[0] * 0.5,
                PAD + n.position[1] + n.size[1] * 0.5,
            ),
            None => return,
        };
        let _ = h
            .input
            .on_cursor_moved(cx, cy, h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();

        let _ = h.input.on_left_pressed(h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();

        // Release so subsequent captures aren't stuck in pressed state.
        let _ = h.input.on_left_released(h.hits, &h.ctx.tree);
        h.react(Instant::now());
        h.settle();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_GLASS").is_some() {
        let new_x = blob_x.get() + 120.0;
        blob_x.set(new_x);
        if let Some(blob) = h.ctx.node("blob") {
            h.ctx.tree.set_position(blob, [new_x, BLOB_Y]);
        }
        h.flush();
        h.render();
        h.capture();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_ANIM").is_some() {
        sigs.hover.set(true);
        h.react(Instant::now());
        let t0 = Instant::now();
        for step in 1..=6u32 {
            let sim = t0 + Duration::from_millis(step as u64 * 40);
            let res = h.timeline.tick(sim);
            if res.updated {
                h.react(Instant::now());
            }
            h.flush();
            h.render();
            h.capture();
        }
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_TOGGLE").is_some() {
        sigs.lit.set(true);
        h.react(Instant::now());
        h.settle();
        h.flush();
        h.render();
        h.capture();
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_OVERDRAW").is_some() {
        h.gpu.set_overdraw(true);
        h.flush();
        h.render();
        h.capture();
        h.gpu.set_overdraw(false);
    }
    if env::var_os("FROSTIFY_AUTOCAPTURE_HUD").is_some() {
        h.show_hud();
        h.render();
        h.capture();
        h.hide_hud();
    }
}

/// Tiny `--capture frames=N out=DIR` parser. Returns `None` when the
/// flag isn't present so callers can fall back to env-var capture or
/// interactive mode. Stage-1 stays clap-free.
fn parse_capture_cli() -> Option<(u32, std::path::PathBuf)> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg != "--capture" {
            continue;
        }
        let mut frames: u32 = 1;
        let mut out = std::path::PathBuf::from("debug_captures");
        for kv in args.by_ref() {
            if kv.starts_with("--") {
                break;
            }
            if let Some(v) = kv.strip_prefix("frames=") {
                frames = v.parse().unwrap_or(1).max(1);
            } else if let Some(v) = kv.strip_prefix("out=") {
                out = std::path::PathBuf::from(v);
            }
        }
        return Some((frames, out));
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let sigs = Sigs::new();
    let blob_x = Rc::new(Cell::new(BLOB_X0));

    let scene_sigs = sigs.clone();
    let key_sigs = sigs.clone();
    let key_blob_x = Rc::clone(&blob_x);

    let mut app = App::new("frostify-gfx", W as u32, H as u32)
        .scene(move |scene| build_scene(scene, &scene_sigs))
        // Honors FROSTIFY_AUTOCAPTURE for plain single-frame CI runs.
        // Multi-frame scripted capture flows attach `.headless(...)`
        // below to drive the demo-specific sub-sequences.
        .capture_from_env();
    if let Some((frames, dir)) = parse_capture_cli() {
        app = app.capture(frames, dir);
    }
    let mut app = app
        .on_key(move |code, state, ctx| {
            if state != ElementState::Pressed {
                return;
            }
            match code {
                KeyCode::Space => {
                    key_sigs.lit.set(!key_sigs.lit.get());
                }
                KeyCode::ArrowLeft | KeyCode::ArrowRight => {
                    let delta = if code == KeyCode::ArrowLeft { -20.0 } else { 20.0 };
                    let new_x = key_blob_x.get() + delta;
                    key_blob_x.set(new_x);
                    if let Some(blob) = ctx.node("blob") {
                        ctx.tree.set_position(blob, [new_x, BLOB_Y]);
                    }
                }
                _ => {}
            }
        });

    if env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
        let hsigs = sigs.clone();
        let hblob = Rc::clone(&blob_x);
        app = app.headless(move |h| run_headless(h, hsigs, hblob));
    }

    app.run()
}
