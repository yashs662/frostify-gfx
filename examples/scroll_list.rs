//! Scroll smoke test. Outer column scrolls vertically through 50
//! coloured rows; an inner row mid-list scrolls horizontally through
//! a wide strip of cards; a glass panel pinned half-way down stays
//! visually correct as content slides under it.
//!
//! Run with:
//!     cargo run --example scroll_list
//!
//! Use the mouse wheel to scroll. Hold Shift while scrolling over the
//! inner row to nudge it horizontally without leaving the row.

use frostify_gfx::{App, BarSide, Justify, Len, Scene};

const W: u32 = 540;
const H: u32 = 540;

fn hsv(h: f32, s: f32, v: f32) -> [f32; 4] {
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

fn build(s: &mut Scene) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .child(|root| {
            // Outer scroll-y list. Custom style: thicker bar pinned to
            // the left, sky-blue thumb that brightens on hover/drag,
            // always visible.
            root.col("list")
                .w(Len::Fill)
                .h(Len::Fill)
                .pad(12.0)
                .gap(8.0)
                .scroll_y()
                .scrollbar(|s| {
                    s.thickness(10.0)
                        .min_thumb(40.0)
                        .margin(6.0)
                        .radius(5.0)
                        .y_side(BarSide::Start)
                        .always_visible(true)
                        .track_color([0.10, 0.12, 0.18, 0.40])
                        .thumb_color([0.40, 0.65, 1.00, 0.55])
                        .thumb_hover_color([0.55, 0.80, 1.00, 0.85])
                        .thumb_active_color([0.85, 0.95, 1.00, 1.00])
                })
                .child(|list| {
                    for i in 0..50u32 {
                        if i == 18 {
                            // Halfway-ish: inner horizontal scroller.
                            list.row("strip")
                                .w(Len::Fill)
                                .h_px(56.0)
                                .pad_xy(8.0, 8.0)
                                .gap(8.0)
                                .rgba(0.10, 0.11, 0.14, 1.0)
                                .radius(8.0)
                                .scroll_x()
                                .scrollbar(|s| {
                                    // Auto-hide: only shows when the
                                    // pointer hovers the bar region or
                                    // a drag is active.
                                    s.thickness(6.0)
                                        .auto_hide(true)
                                        .thumb_color([1.0, 0.85, 0.30, 0.60])
                                        .thumb_hover_color([1.0, 0.92, 0.55, 0.90])
                                })
                                .child(|strip| {
                                    for j in 0..20u32 {
                                        let c = hsv(j as f32 / 20.0, 0.55, 0.95);
                                        strip
                                            .rect(format!("card{j}"))
                                            .w_px(120.0)
                                            .h_px(40.0)
                                            .color(c)
                                            .radius(6.0);
                                    }
                                });
                            continue;
                        }
                        let c = hsv((i as f32 * 0.13).fract(), 0.45, 0.85);
                        list.row(format!("row{i}"))
                            .w(Len::Fill)
                            .h_px(36.0)
                            .pad_xy(12.0, 6.0)
                            .gap(8.0)
                            .justify(Justify::Start)
                            .color(c)
                            .radius(6.0)
                            .child(|r| {
                                r.text(format!("row{i}_label"), format!("row {i:>02}"), 14.0)
                                    .color([0.0, 0.0, 0.0, 0.85]);
                            });
                    }
                });
            // Glass panel pinned over the list — content scrolls
            // visibly behind it.
            root.glass("panel")
                .abs(60.0, 220.0)
                .size_px(W as f32 - 120.0, 100.0)
                .radius(16.0)
                .blur(20.0)
                .refraction(8.0)
                .roughness(1.0)
                .rgba(1.0, 1.0, 1.0, 0.08);
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let app = App::new("scroll list", W, H).scene(build);
    app.run()
}
