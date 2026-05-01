//! Rgba8UnormSrgb image atlas.
//!
//! Backed by a single square `Rgba8UnormSrgb` texture and an `etagere`
//! shelf-pack allocator. Keyed on a monotonic [`ImageHandle`]; users
//! upload PNG bytes via [`ImageAtlas::upload_png`] and stash the handle
//! to reference the image in scene nodes.
//!
//! When the atlas fills up [`ImageAtlas::upload_png`] returns `None` —
//! stage-1 has no eviction. `Rgba8UnormSrgb` matches the shape pipeline
//! convention: PNG bytes are sRGB-authored, the texture decode flag
//! converts to linear when sampled, fragment math stays linear, the
//! swapchain encodes back to sRGB on store.

use std::collections::HashMap;
use std::io::Cursor;

use etagere::{size2, AtlasAllocator};

/// Opaque handle returned by [`ImageAtlas::upload_png`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub u32);

/// UV + pixel size for an uploaded image.
#[derive(Copy, Clone, Debug)]
pub struct ImageEntry {
    /// UV rect in `[0, 1]^2` — `[u0, v0, u1, v1]`.
    pub uv: [f32; 4],
    pub width: u32,
    pub height: u32,
}

pub struct ImageAtlas {
    size: u32,
    texture: wgpu::Texture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    allocator: AtlasAllocator,
    occupants: HashMap<ImageHandle, ImageEntry>,
    next_handle: u32,
    /// Transparent border around each image to keep linear filtering
    /// from bleeding across neighbours.
    padding: i32,
}

impl ImageAtlas {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify-gfx image atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify-gfx image sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frostify-gfx image atlas bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frostify-gfx image atlas bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let allocator = AtlasAllocator::new(size2(size as i32, size as i32));
        Self {
            size,
            texture,
            layout,
            bind_group,
            allocator,
            occupants: HashMap::new(),
            next_handle: 0,
            padding: 1,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn get(&self, handle: ImageHandle) -> Option<ImageEntry> {
        self.occupants.get(&handle).copied()
    }

    /// Decode a PNG byte slice and upload its pixels into the atlas.
    /// Returns `None` if the PNG is malformed, larger than the atlas,
    /// or the atlas is full. RGBA8 internally; sRGB inputs welcome.
    pub fn upload_png(
        &mut self,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> Option<ImageHandle> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().ok()?;
        let (w, h, color_type, bit_depth, buf_size) = {
            let info = reader.info();
            (info.width, info.height, info.color_type, info.bit_depth, reader.output_buffer_size()?)
        };
        let mut buf = vec![0u8; buf_size];
        let frame = reader.next_frame(&mut buf).ok()?;
        let in_bytes = &buf[..frame.buffer_size()];
        // Normalize to RGBA8.
        let rgba = match (color_type, bit_depth) {
            (png::ColorType::Rgba, png::BitDepth::Eight) => in_bytes.to_vec(),
            (png::ColorType::Rgb, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for px in in_bytes.chunks_exact(3) {
                    out.extend_from_slice(&[px[0], px[1], px[2], 0xFF]);
                }
                out
            }
            (png::ColorType::Grayscale, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for &g in in_bytes {
                    out.extend_from_slice(&[g, g, g, 0xFF]);
                }
                out
            }
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for px in in_bytes.chunks_exact(2) {
                    out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                }
                out
            }
            _ => return None,
        };
        self.upload_rgba(queue, w, h, &rgba)
    }

    /// Upload pre-decoded `Rgba8UnormSrgb` pixels (`w*h*4` bytes,
    /// row-major, top-left origin). Stricter than [`upload_png`] —
    /// caller is responsible for color-space correctness.
    pub fn upload_rgba(
        &mut self,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        rgba: &[u8],
    ) -> Option<ImageHandle> {
        if w == 0 || h == 0 || rgba.len() != (w * h * 4) as usize {
            return None;
        }
        let pad = self.padding;
        let pad_w = w as i32 + 2 * pad;
        let pad_h = h as i32 + 2 * pad;
        if pad_w > self.size as i32 || pad_h > self.size as i32 {
            return None;
        }
        let alloc = self.allocator.allocate(size2(pad_w, pad_h))?;
        let rect = alloc.rectangle;
        let gx = rect.min.x + pad;
        let gy = rect.min.y + pad;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: gx as u32,
                    y: gy as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let inv = 1.0 / self.size as f32;
        let entry = ImageEntry {
            uv: [
                gx as f32 * inv,
                gy as f32 * inv,
                (gx + w as i32) as f32 * inv,
                (gy + h as i32) as f32 * inv,
            ],
            width: w,
            height: h,
        };
        let handle = ImageHandle(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1);
        let _ = alloc.id;
        self.occupants.insert(handle, entry);
        Some(handle)
    }

    /// Reported GPU bytes used by the atlas texture.
    pub fn memory_bytes(&self) -> u64 {
        self.size as u64 * self.size as u64 * 4
    }
}
