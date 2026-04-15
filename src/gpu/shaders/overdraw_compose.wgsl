// Overdraw heatmap — compose pass.
//
// Fullscreen quad samples the per-pixel shape count and remaps it through
// a 5-stop colour ramp (black → blue → cyan → yellow → red).

@group(0) @binding(0) var overdraw_tex: texture_2d<f32>;
@group(0) @binding(1) var overdraw_samp: sampler;

struct ComposeOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_compose(@builtin(vertex_index) vi: u32) -> ComposeOut {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    var out: ComposeOut;
    out.pos = vec4<f32>(corner * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    // The count texture was authored with world (0,0) → texture (0,0)
    // (top-left). Flip Y here so screen-bottom samples texture-bottom.
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

fn heatmap(t: f32) -> vec3<f32> {
    // 5-stop ramp: black → blue → cyan → yellow → red.
    let s0 = vec3<f32>(0.0, 0.0, 0.0);
    let s1 = vec3<f32>(0.0, 0.0, 0.6);
    let s2 = vec3<f32>(0.0, 0.8, 0.8);
    let s3 = vec3<f32>(1.0, 0.9, 0.0);
    let s4 = vec3<f32>(1.0, 0.1, 0.0);
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let i = u32(floor(x));
    let f = fract(x);
    var lo: vec3<f32>;
    var hi: vec3<f32>;
    switch i {
        case 0u: { lo = s0; hi = s1; }
        case 1u: { lo = s1; hi = s2; }
        case 2u: { lo = s2; hi = s3; }
        default: { lo = s3; hi = s4; }
    }
    return mix(lo, hi, f);
}

@fragment
fn fs_compose(in: ComposeOut) -> @location(0) vec4<f32> {
    let count = textureSampleLevel(overdraw_tex, overdraw_samp, in.uv, 0.0).r;
    // Normalize against 8 overlapping shapes — anything past saturates red.
    let t = count / 8.0;
    let rgb = heatmap(t);
    return vec4<f32>(rgb, 1.0);
}
