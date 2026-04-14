// Separable Gaussian blur compute pass.
//
// Run this shader twice per frame: once with direction=(1,0), once with
// (0,1), ping-ponging between two rgba8unorm textures. Kernel radius is
// a runtime uniform so the CPU can scale spread to the viewport.

struct BlurParams {
    // (1,0) horizontal, (0,1) vertical.
    direction: vec2<i32>,
    radius: u32,
    _pad: u32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(dst));
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }

    let r = i32(params.radius);
    // sigma chosen so ~3 sigma covers the kernel radius.
    let sigma = max(f32(r) * (1.0 / 3.0), 1.0);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    var accum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let offset = params.direction * i;
        let cc = clamp(coord + offset, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        let fi = f32(i);
        let w = exp(-fi * fi * inv_two_sigma_sq);
        accum = accum + textureLoad(src, cc, 0) * w;
        weight_sum = weight_sum + w;
    }

    textureStore(dst, coord, accum / weight_sum);
}
