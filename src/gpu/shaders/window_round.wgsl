// Final window-corner round pass.
//
// The whole frame is composited into an offscreen texture, then this pass
// blits it to the swapchain while masking the four window corners to a
// rounded rect. Rounding the *composited* result once (rather than each
// instance/layer during raster) means stacked translucent content — e.g. the
// ambient glass over the backdrop art — can't leak the back layer through a
// per-layer anti-aliased corner: there is a single corner boundary.
//
// Straight edges are left fully covered (they coincide with the surface
// boundary, which the compositor clips) — only the corner arcs feather.

struct RoundU {
    // Surface size in physical px.
    screen: vec2<f32>,
    // Corner radius in physical px. 0 = no rounding (straight blit).
    radius: f32,
    _pad: f32,
}

@group(0) @binding(0) var frame_tex: texture_2d<f32>;
@group(0) @binding(1) var frame_samp: sampler;
@group(0) @binding(2) var<uniform> u: RoundU;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = p[vi];
    var out: VSOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // UV: clip → [0,1], y flipped (texture origin top-left).
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return out;
}

// Coverage of the rounded window: 1.0 everywhere except inside the four
// corner boxes, where a circular arc anti-aliases the cut. Straight edges
// stay hard (fully covered).
fn corner_coverage(px: vec2<f32>) -> f32 {
    let r = u.radius;
    if (r <= 0.0) {
        return 1.0;
    }
    let half = u.screen * 0.5;
    let p = abs(px - half);
    let corner = half - vec2<f32>(r);
    if (p.x <= corner.x || p.y <= corner.y) {
        return 1.0;
    }
    let d = length(p - corner) - r;
    let aa = max(fwidth(d), 0.5);
    return smoothstep(aa, -aa, d);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    // The composited frame is premultiplied alpha; scaling by coverage keeps
    // it premultiplied.
    let s = textureSampleLevel(frame_tex, frame_samp, in.uv, 0.0);
    return s * corner_coverage(in.pos.xy);
}
