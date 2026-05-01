//! Layering smoke test. Builds a scene with text, image, glass and
//! plain rects declared in arbitrary order and captures one frame to
//! `debug_captures/layering/`. The capture should show:
//!
//!   - bottom row: a magenta rect overlapping a frosted glass panel,
//!     with text and an image declared **behind** the glass — both
//!     should appear blurred through the glass.
//!   - on top of the glass: a label declared **after** the glass,
//!     drawn crisp.
//!
//! Run with:
//!     cargo run --example layering

use frostify_gfx::{App, Justify, Len, Scene};

const W: u32 = 720;
const H: u32 = 420;

fn make_demo_image() -> (u32, u32, Vec<u8>) {
    const WI: u32 = 96;
    const HI: u32 = 96;
    let mut bytes = Vec::with_capacity((WI * HI * 4) as usize);
    for y in 0..HI {
        for x in 0..WI {
            let fx = x as f32 / (WI - 1) as f32;
            let fy = y as f32 / (HI - 1) as f32;
            let band = ((x / 8) + (y / 8)) % 2 == 0;
            let r = (fx * 255.0) as u8;
            let g = ((1.0 - fy) * 255.0) as u8;
            let b = if band { 240 } else { 60 };
            bytes.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (WI, HI, bytes)
}

fn build(s: &mut Scene, art: frostify_gfx::ImageHandle) {
    s.col("root")
        .fill()
        .pad(20.0)
        .gap(12.0)
        .rgba(0.05, 0.06, 0.08, 1.0)
        .child(|p| {
            // Header — text on a colored bar.
            p.row("hdr")
                .w(Len::Fill)
                .h_px(48.0)
                .pad(12.0)
                .gap(10.0)
                .justify(Justify::Start)
                .rgba(0.13, 0.14, 0.18, 1.0)
                .radius(10.0)
                .child(|h| {
                    h.text("hdr_label", "layering smoke test", 18.0)
                        .color([1.0, 1.0, 1.0, 0.95]);
                });
            // Stage. Each child is absolutely positioned; declared
            // order = paint order.
            p.col("stage")
                .w(Len::Fill)
                .h(Len::Fill)
                .rgba(0.18, 0.20, 0.24, 1.0)
                .radius(10.0)
                .child(|c| {
                    // Layer 1 (back): magenta rect.
                    c.rect("back_rect")
                        .abs(40.0, 40.0)
                        .size_px(320.0, 200.0)
                        .rgba(0.95, 0.20, 0.55, 1.0)
                        .radius(18.0);
                    // Layer 2: image over the rect.
                    c.image("art", art).abs(120.0, 70.0).size_px(96.0, 96.0).radius(12.0);
                    // Layer 3: text over the image. Should appear
                    // blurred through the glass below.
                    c.text("under_glass", "BEHIND GLASS", 28.0)
                        .abs(60.0, 200.0)
                        .color([1.0, 1.0, 1.0, 1.0]);
                    // Layer 4: glass — blurs everything declared above.
                    c.glass("panel")
                        .abs(40.0, 130.0)
                        .size_px(420.0, 130.0)
                        .radius(20.0)
                        .blur(24.0)
                        .refraction(10.0)
                        .rgba(1.0, 1.0, 1.0, 0.10);
                    // Layer 5: text over the glass. Crisp.
                    c.text("over_glass", "in front of glass", 22.0)
                        .abs(60.0, 175.0)
                        .color([1.0, 1.0, 1.0, 1.0]);
                });
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let mut app = App::new("layering smoke", W, H);
    let (iw, ih, bytes) = make_demo_image();
    let art = app.stage_image_rgba(iw, ih, bytes);
    let app = app
        .scene(move |s| build(s, art))
        .capture(1, "debug_captures/layering");
    app.run()
}
