// SDF pipeline for M4 shapes: rounded rectangles with per-corner radii,
// borders, drop shadows, plus frosted-glass shapes that sample a
// pre-blurred backdrop texture. Outputs PREMULTIPLIED alpha — pair with
// blend state One / OneMinusSrcAlpha and a PreMultiplied surface
// composite mode.
//
// Two fragment entry points:
//   - `fs_opaque`: no @group(1) usage. Used by the backdrop pass (which
//     targets an offscreen rgba8unorm and must not sample the texture it
//     is writing).
//   - `fs_main`: includes the glass branch that samples `blurred_tex`.
//     Used by the final surface pass.

const SHAPE_KIND_RECT: u32  = 0u;
const SHAPE_KIND_GLASS: u32 = 1u;

struct ShapeInstance {
    color: vec4<f32>,
    border_color: vec4<f32>,
    shadow_color: vec4<f32>,
    border_radius: vec4<f32>,   // TL, TR, BL, BR (pixels)
    backdrop_uv_rect: vec4<f32>,
    position: vec2<f32>,        // top-left in pixels
    size: vec2<f32>,            // pixels
    shadow_offset: vec2<f32>,
    shape_kind: u32,
    roughness: f32,
    border_width: f32,
    shadow_blur: f32,
    shadow_opacity: f32,
    opacity: f32,
}

struct Frame {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;

// Final-pass-only bindings. The opaque-pass pipeline layout omits group 1
// so only fs_main references these.
@group(1) @binding(0) var blurred_tex: texture_2d<f32>;
@group(1) @binding(1) var blurred_samp: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) inst_idx: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VSOut {
    let inst = instances[ii];

    // Expand quad to cover dropped shadow so fragment shader sees enough area.
    let shadow_pad = vec2<f32>(
        abs(inst.shadow_offset.x) + max(inst.shadow_blur, 0.0) * 2.0,
        abs(inst.shadow_offset.y) + max(inst.shadow_blur, 0.0) * 2.0,
    );
    let min_px = inst.position - shadow_pad;
    let max_px = inst.position + inst.size + shadow_pad;

    // Two triangles, six verts. vertex_index -> unit-square corner.
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let world_pos = mix(min_px, max_px, corner);

    // Pixel -> clip-space. Y flipped so (0,0) is top-left.
    let ndc = vec2<f32>(
        world_pos.x / frame.screen_size.x * 2.0 - 1.0,
        1.0 - world_pos.y / frame.screen_size.y * 2.0,
    );

    var out: VSOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.world_pos = world_pos;
    out.inst_idx = ii;
    return out;
}

// Decode an sRGB-authored color component to linear. ShapeInstance colors
// are authored in perceptual sRGB space (the numbers a designer types), so
// the shader converts once at fragment entry and does all math/blending in
// linear space. The swapchain is Bgra8UnormSrgb which auto-encodes the
// linear output back to sRGB bytes on store.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(lo, hi, c > cutoff);
}

fn sd_rectangle(p: vec2<f32>, xy: vec2<f32>) -> f32 {
    let d = abs(p) - max(xy, vec2<f32>(0.0));
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sd_rounded_rect(p: vec2<f32>, xy: vec2<f32>, r: vec4<f32>) -> f32 {
    let qx = select(0u, 1u, p.x > 0.0);
    let qy = select(0u, 2u, p.y < 0.0);
    let idx = qx + qy;
    var s: f32;
    switch idx {
        case 0u: { s = r.x; }      // top-left
        case 1u: { s = r.y; }      // top-right
        case 2u: { s = r.z; }      // bottom-left
        default: { s = r.w; }      // bottom-right
    }
    s = min(s, min(xy.x, xy.y));
    return sd_rectangle(p, xy - vec2<f32>(s)) - s;
}

struct Body { rgb: vec3<f32>, a: f32 };

// Rect / border fill — shared between both fragment entry points.
// Color channels are sRGB-decoded to linear before any blending math.
fn compute_fill(inst: ShapeInstance, p: vec2<f32>, outer_half: vec2<f32>,
                comp_alpha: f32, aa: f32) -> Body {
    var body: Body;
    let color_lin = srgb_to_linear(inst.color.rgb);
    if (inst.border_width > 0.0) {
        let border_lin = srgb_to_linear(inst.border_color.rgb);
        let inner_half = max(outer_half - vec2<f32>(inst.border_width), vec2<f32>(0.0));
        let inner_radii = max(inst.border_radius - vec4<f32>(inst.border_width), vec4<f32>(0.0));
        let inner_dist = sd_rounded_rect(p, inner_half, inner_radii);
        let inner_alpha = smoothstep(aa, -aa, inner_dist);
        let border_alpha = clamp(comp_alpha - inner_alpha, 0.0, 1.0);
        let fill_a = inst.color.a * inner_alpha;
        let fill_rgb = color_lin * fill_a;
        let bord_a = inst.border_color.a * border_alpha;
        let bord_rgb = border_lin * bord_a;
        body.a = bord_a + fill_a * (1.0 - bord_a);
        body.rgb = bord_rgb + fill_rgb * (1.0 - bord_a);
    } else {
        body.a = inst.color.a * comp_alpha;
        body.rgb = color_lin * body.a;
    }
    return body;
}

// Shadow contribution — shared between both entry points.
fn compute_shadow(inst: ShapeInstance, px: vec2<f32>, center: vec2<f32>,
                  outer_half: vec2<f32>) -> Body {
    var s: Body;
    s.rgb = vec3<f32>(0.0);
    s.a = 0.0;
    if (inst.shadow_opacity > 0.0) {
        let sd = sd_rounded_rect(px - (center + inst.shadow_offset),
                                 outer_half, inst.border_radius);
        let s_edge = max(inst.shadow_blur, 1.0);
        let s_i = smoothstep(s_edge, -s_edge, sd);
        s.a = inst.shadow_color.a * s_i * inst.shadow_opacity;
        s.rgb = srgb_to_linear(inst.shadow_color.rgb) * s.a;
    }
    return s;
}

// Fragment entry for the backdrop (opaque) pass. No texture binding
// references → pipeline layout omits group 1.
@fragment
fn fs_opaque(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;
    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5;
    let comp_dist = sd_rounded_rect(p, outer_half, inst.border_radius);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);

    let body = compute_fill(inst, p, outer_half, comp_alpha, aa);
    let shadow = compute_shadow(inst, px, center, outer_half);

    let out_a = body.a + shadow.a * (1.0 - body.a);
    let out_rgb = body.rgb + shadow.rgb * (1.0 - body.a);
    return vec4<f32>(out_rgb, out_a) * inst.opacity;
}

// Fragment entry for the final surface pass. Handles glass by sampling
// the pre-blurred backdrop texture.
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let inst = instances[in.inst_idx];
    let px = in.world_pos;
    let center = inst.position + inst.size * 0.5;
    let p = px - center;
    let outer_half = inst.size * 0.5;
    let comp_dist = sd_rounded_rect(p, outer_half, inst.border_radius);
    let aa = max(fwidth(comp_dist), 0.5);
    let comp_alpha = smoothstep(aa, -aa, comp_dist);

    var body: Body;
    if (inst.shape_kind == SHAPE_KIND_GLASS) {
        // Screen-space UV — blurred_tex is full-res and aligned to the surface.
        let uv = px / frame.screen_size;
        // Tiny per-pixel extra spread driven by roughness. Cheap 5-tap.
        let spread = max(inst.roughness, 0.0) * 2.0 / frame.screen_size;
        let bg = (
            textureSampleLevel(blurred_tex, blurred_samp, uv, 0.0) * 2.0 +
            textureSampleLevel(blurred_tex, blurred_samp, uv + vec2<f32>(spread.x, 0.0), 0.0) +
            textureSampleLevel(blurred_tex, blurred_samp, uv - vec2<f32>(spread.x, 0.0), 0.0) +
            textureSampleLevel(blurred_tex, blurred_samp, uv + vec2<f32>(0.0, spread.y), 0.0) +
            textureSampleLevel(blurred_tex, blurred_samp, uv - vec2<f32>(0.0, spread.y), 0.0)
        ) * (1.0 / 6.0);
        // `bg` is already premultiplied and already linear: the opaque
        // pass wrote sRGB-decoded colors into the rgba8unorm backdrop
        // target, so sampling returns linear values directly.
        let tint_a = inst.color.a;
        let tint_lin = srgb_to_linear(inst.color.rgb);
        let tint_rgb_p = tint_lin * tint_a;
        let fill_rgb_p = tint_rgb_p + bg.rgb * (1.0 - tint_a);
        let fill_a = tint_a + bg.a * (1.0 - tint_a);
        // Mask by SDF alpha so rounded corners clip cleanly.
        body.rgb = fill_rgb_p * comp_alpha;
        body.a = fill_a * comp_alpha;
    } else {
        body = compute_fill(inst, p, outer_half, comp_alpha, aa);
    }

    let shadow = compute_shadow(inst, px, center, outer_half);
    let out_a = body.a + shadow.a * (1.0 - body.a);
    let out_rgb = body.rgb + shadow.rgb * (1.0 - body.a);
    return vec4<f32>(out_rgb, out_a) * inst.opacity;
}
