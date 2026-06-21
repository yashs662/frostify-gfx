//! Backdrop capture + mipmap pyramid for per-glass blur.
//!
//! The opaque pass writes mip 0 of `backdrop_tex`. After the pass, the
//! `cs_downsample` compute shader runs once per mip transition to fill
//! mips 1..N as 4-tap bilinear averages of the previous mip. The glass
//! shader then samples the pyramid via `textureSampleLevel` with a
//! fractional LOD = `log2(blur_px)` — the sampler's `mipmap_filter:
//! Linear` interpolates between adjacent mip levels for smooth
//! variable-radius blur.

pub const BACKDROP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Cap on pyramid depth. log2(2048)+1 = 12, but realistically 8 levels
/// already cover blur radii up to 256 px; capping keeps allocation
/// counts predictable across resizes.
const MAX_MIPS: u32 = 8;

pub struct BlurResources {
    pub backdrop_tex: wgpu::Texture,
    /// View covering all mips (mip 0..N-1). Bound by the glass branch
    /// of the final pass; sampled with fractional LOD.
    pub backdrop_view: wgpu::TextureView,
    /// Single-mip view at level 0. Used as the render-attachment view
    /// for the opaque pass — wgpu rejects a multi-mip view as a render
    /// target.
    pub backdrop_mip0_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,

    /// Number of mip levels actually allocated for the current size.
    /// Used by the shader to clamp `log2(blur_px)` to a valid LOD.
    mip_count: u32,
    /// One bind group per downsample step. `bg[i]` reads mip `i` and
    /// writes mip `i+1`. `bg.len() == mip_count - 1`.
    downsample_bgs: Vec<wgpu::BindGroup>,

    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bgl: wgpu::BindGroupLayout,
    /// Sampler used by the downsample shader. Bilinear, no mip filter
    /// — every read hits a single mip level.
    downsample_samp: wgpu::Sampler,

    width: u32,
    height: u32,
}

impl BlurResources {
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_count = mip_count_for(width, height);
        let (backdrop_tex, backdrop_view) =
            make_tex(device, width, height, mip_count, "opal.backdrop");
        let backdrop_mip0_view = make_mip0_view(&backdrop_tex);

        // Sampler used by glass shapes. Linear mipmap filter is
        // load-bearing: it's what turns `textureSampleLevel(tex, samp,
        // uv, fractional_lod)` into a trilinear blend between adjacent
        // mip levels.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("opal.backdrop sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let downsample_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("opal.downsample sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.downsample shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/downsample.wgsl").into()),
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal.downsample bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: BACKDROP_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.downsample pl"),
            bind_group_layouts: &[Some(&downsample_bgl)],
            immediate_size: 0,
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("opal.downsample pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let downsample_bgs = build_downsample_bgs(
            device,
            &downsample_bgl,
            &downsample_samp,
            &backdrop_tex,
            mip_count,
        );

        Self {
            backdrop_tex,
            backdrop_view,
            backdrop_mip0_view,
            sampler,
            mip_count,
            downsample_bgs,
            downsample_pipeline,
            downsample_bgl,
            downsample_samp,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let mip_count = mip_count_for(width, height);
        let (tex, view) = make_tex(device, width, height, mip_count, "opal.backdrop");
        let mip0 = make_mip0_view(&tex);
        self.backdrop_tex = tex;
        self.backdrop_view = view;
        self.backdrop_mip0_view = mip0;
        self.mip_count = mip_count;
        self.downsample_bgs = build_downsample_bgs(
            device,
            &self.downsample_bgl,
            &self.downsample_samp,
            &self.backdrop_tex,
            mip_count,
        );
        self.width = width;
        self.height = height;
    }

    /// Run `mip_count - 1` downsample dispatches: mip0→mip1, mip1→mip2,
    /// etc. Caller is responsible for ensuring mip 0 has been written
    /// (e.g. by the opaque render pass) before invoking.
    pub fn run_downsample(&self, encoder: &mut wgpu::CommandEncoder) {
        for i in 0..self.downsample_bgs.len() {
            let dst_w = (self.width >> (i + 1)).max(1);
            let dst_h = (self.height >> (i + 1)).max(1);
            let gx = dst_w.div_ceil(8);
            let gy = dst_h.div_ceil(8);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("opal.downsample"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.downsample_pipeline);
            cpass.set_bind_group(0, &self.downsample_bgs[i], &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
    }
}

/// `floor(log2(max_dim))` clamped to `[1, MAX_MIPS]`. With surface
/// 1100×750 this yields 8.
fn mip_count_for(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height).max(1);
    let bits = 32 - max_dim.leading_zeros();
    bits.clamp(1, MAX_MIPS)
}

fn make_mip0_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("opal.backdrop.mip0"),
        base_mip_level: 0,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

fn make_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    mip_count: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BACKDROP_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // Default view spans all mips — used for sampling from the glass shader.
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn build_downsample_bgs(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    tex: &wgpu::Texture,
    mip_count: u32,
) -> Vec<wgpu::BindGroup> {
    let mut out = Vec::with_capacity(mip_count.saturating_sub(1) as usize);
    for i in 0..mip_count.saturating_sub(1) {
        let src_view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("opal.downsample.src"),
            base_mip_level: i,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("opal.downsample.dst"),
            base_mip_level: i + 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opal.downsample bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });
        out.push(bg);
    }
    out
}
