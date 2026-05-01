use bytemuck::{Pod, Zeroable};

/// GPU shape kinds. Mirror the constants in `shape.wgsl`.
pub const SHAPE_KIND_RECT: u32 = 0;
pub const SHAPE_KIND_GLASS: u32 = 1;
/// Glyph blit: samples the `R8Unorm` glyph atlas at
/// `backdrop_uv_rect` (repurposed as atlas `(u0, v0, w, h)`),
/// multiplied by `color`. Shadows/borders disabled; `position`+`size`
/// bound the bitmap quad exactly.
pub const SHAPE_KIND_GLYPH: u32 = 2;
/// Image blit: samples the `Rgba8UnormSrgb` image atlas at
/// `backdrop_uv_rect` (repurposed as atlas `(u0, v0, w, h)`),
/// multiplied by `color` (used as tint; `[1,1,1,1]` for unmodified).
pub const SHAPE_KIND_IMAGE: u32 = 3;

/// GPU shape instance. Layout must match `ShapeInstance` in `shape.wgsl`.
/// std430-compatible. **128 bytes, 16-aligned.** WGSL `array<ShapeInstance>`
/// stride = roundUp(16, last_offset + last_size) = 128. The Rust struct
/// size must match exactly — do not add trailing pad fields (the
/// alignment-rounded stride bites silently otherwise).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShapeInstance {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub shadow_color: [f32; 4],
    pub border_radius: [f32; 4], // tl, tr, bl, br
    /// Multi-purpose by `shape_kind`:
    ///   - Glass: `(blur_px, refraction_px, _, _)`
    ///   - Glyph: atlas `(u0, v0, w, h)` into the R8 glyph atlas
    ///   - Image: atlas `(u0, v0, w, h)` into the Rgba8 image atlas
    ///   - Rect: ignored
    pub backdrop_uv_rect: [f32; 4],
    pub position: [f32; 2], // top-left, pixels
    pub size: [f32; 2],     // pixels
    pub shadow_offset: [f32; 2],
    pub shape_kind: u32,
    /// Reserved for future per-instance use; keeps the struct at 128 B
    /// so `array<ShapeInstance>` stride matches WGSL.
    pub _pad0: f32,
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
            _pad0: 0.0,
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

/// Per-frame uniform: viewport size in physical pixels + max usable
/// LOD for the backdrop mip pyramid (so the glass shader can clamp
/// `log2(blur_px)` without overflowing into a 1×1 mip).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub screen_size: [f32; 2],
    pub max_backdrop_lod: f32,
    pub _pad: f32,
}
