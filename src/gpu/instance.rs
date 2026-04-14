use bytemuck::{Pod, Zeroable};

/// GPU shape kinds. Mirror the constants in `shape.wgsl`.
pub const SHAPE_KIND_RECT: u32 = 0;
pub const SHAPE_KIND_GLASS: u32 = 1;

/// GPU shape instance. Layout must match `ShapeInstance` in `shape.wgsl`.
/// std430-compatible. **128 bytes, 16-aligned.** WGSL `array<ShapeInstance>`
/// stride = roundUp(16, last_offset + last_size) = 128. The Rust struct size
/// must match exactly — do not add trailing pad fields (see M2 landmine note
/// in PLAN.md).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShapeInstance {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub shadow_color: [f32; 4],
    pub border_radius: [f32; 4], // tl, tr, bl, br
    /// UV rect into the blurred backdrop texture: (u0, v0, u1-u0, v1-v0).
    /// Only read by SHAPE_KIND_GLASS. Ignored otherwise.
    pub backdrop_uv_rect: [f32; 4],
    pub position: [f32; 2], // top-left, pixels
    pub size: [f32; 2],     // pixels
    pub shadow_offset: [f32; 2],
    pub shape_kind: u32,
    pub roughness: f32,
    pub border_width: f32,
    pub shadow_blur: f32,
    pub shadow_opacity: f32,
    pub opacity: f32,
}

const _: () = assert!(std::mem::size_of::<ShapeInstance>() == 128);

impl Default for ShapeInstance {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            border_radius: [0.0; 4],
            backdrop_uv_rect: [0.0; 4],
            position: [0.0; 2],
            size: [0.0; 2],
            shadow_offset: [0.0; 2],
            shape_kind: SHAPE_KIND_RECT,
            roughness: 0.0,
            border_width: 0.0,
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
        }
    }
}

impl ShapeInstance {
    pub fn rounded_rect(
        position: [f32; 2],
        size: [f32; 2],
        radius: f32,
        color: [f32; 4],
    ) -> Self {
        Self {
            color,
            position,
            size,
            border_radius: [radius; 4],
            ..Default::default()
        }
    }
}

/// Per-frame uniform: viewport size in physical pixels.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub screen_size: [f32; 2],
    pub _pad: [f32; 2],
}
