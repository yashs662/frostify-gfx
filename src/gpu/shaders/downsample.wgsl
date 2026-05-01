// Dual-filter downsample (Bjørge / ARM). 5 bilinear taps weighted as
// (center*4 + 4 corners) / 8 — a closer match to a Gaussian than the
// straight 4-tap box average. Combined with the matching dual-up
// kernel in shape.wgsl's glass branch this approximates a true
// Gaussian blur over the chosen mip depth without the visible
// box-pixel character of repeated naïve box averaging.
//
// Run once per mip transition; `src` = previous mip, `dst` = current
// (half-resolution) mip.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = vec2<i32>(textureDimensions(dst));
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    if (coord.x >= dst_dims.x || coord.y >= dst_dims.y) {
        return;
    }
    let src_dims = vec2<f32>(textureDimensions(src));
    let inv_src = vec2<f32>(1.0) / src_dims;
    let inv_dst = vec2<f32>(1.0) / vec2<f32>(dst_dims);
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) * inv_dst;
    // Half a source-pixel offset on each axis.
    let hp = 0.5 * inv_src;
    let c  = textureSampleLevel(src, src_samp, uv,                                  0.0) * 4.0;
    let tl = textureSampleLevel(src, src_samp, uv + vec2<f32>(-hp.x, -hp.y), 0.0);
    let tr = textureSampleLevel(src, src_samp, uv + vec2<f32>( hp.x, -hp.y), 0.0);
    let bl = textureSampleLevel(src, src_samp, uv + vec2<f32>(-hp.x,  hp.y), 0.0);
    let br = textureSampleLevel(src, src_samp, uv + vec2<f32>( hp.x,  hp.y), 0.0);
    let avg = (c + tl + tr + bl + br) * (1.0 / 8.0);
    textureStore(dst, coord, avg);
}
