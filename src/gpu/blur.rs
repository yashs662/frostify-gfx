//! Backdrop capture + separable Gaussian blur resources.
//!
//! Owns three textures:
//!   - `backdrop`: opaque pass renders here (also sampled by the horizontal
//!     blur pass).
//!   - `tmp`: ping target for the horizontal pass (sampled by vertical).
//!   - `blurred`: final vertical-pass output, sampled by glass shapes in
//!     the surface pass.
//!
//! All three are `rgba8unorm` so they're valid storage-image targets for
//! the compute pipeline. Recreated on resize.

use bytemuck::{Pod, Zeroable};

pub const BACKDROP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlurParams {
    direction: [i32; 2],
    radius: u32,
    _pad: u32,
}

pub struct BlurResources {
    pub backdrop_tex: wgpu::Texture,
    pub backdrop_view: wgpu::TextureView,
    pub tmp_tex: wgpu::Texture,
    pub tmp_view: wgpu::TextureView,
    pub blurred_tex: wgpu::Texture,
    pub blurred_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,

    compute_pipeline: wgpu::ComputePipeline,
    compute_bgl: wgpu::BindGroupLayout,
    params_h: wgpu::Buffer,
    params_v: wgpu::Buffer,
    // BindGroups are recreated on resize alongside textures.
    bg_h: wgpu::BindGroup,
    bg_v: wgpu::BindGroup,

    width: u32,
    height: u32,
}

impl BlurResources {
    /// Current offscreen resolution of the three blur textures.
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl BlurResources {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (backdrop_tex, backdrop_view) = make_tex(device, width, height, "frostify.backdrop");
        let (tmp_tex, tmp_view) = make_tex(device, width, height, "frostify.blur.tmp");
        let (blurred_tex, blurred_view) =
            make_tex(device, width, height, "frostify.blur.blurred");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frostify.blur sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frostify.blur shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frostify.blur bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: BACKDROP_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<BlurParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frostify.blur pl"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frostify.blur pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.blur params.h"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.blur params.v"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_h = make_bg(
            device,
            &compute_bgl,
            &backdrop_view,
            &tmp_view,
            &params_h,
            "frostify.blur bg.h",
        );
        let bg_v = make_bg(
            device,
            &compute_bgl,
            &tmp_view,
            &blurred_view,
            &params_v,
            "frostify.blur bg.v",
        );

        Self {
            backdrop_tex,
            backdrop_view,
            tmp_tex,
            tmp_view,
            blurred_tex,
            blurred_view,
            sampler,
            compute_pipeline,
            compute_bgl,
            params_h,
            params_v,
            bg_h,
            bg_v,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let (b, bv) = make_tex(device, width, height, "frostify.backdrop");
        let (t, tv) = make_tex(device, width, height, "frostify.blur.tmp");
        let (r, rv) = make_tex(device, width, height, "frostify.blur.blurred");
        self.backdrop_tex = b;
        self.backdrop_view = bv;
        self.tmp_tex = t;
        self.tmp_view = tv;
        self.blurred_tex = r;
        self.blurred_view = rv;
        self.bg_h = make_bg(
            device,
            &self.compute_bgl,
            &self.backdrop_view,
            &self.tmp_view,
            &self.params_h,
            "frostify.blur bg.h",
        );
        self.bg_v = make_bg(
            device,
            &self.compute_bgl,
            &self.tmp_view,
            &self.blurred_view,
            &self.params_v,
            "frostify.blur bg.v",
        );
        self.width = width;
        self.height = height;
    }

    /// Encode both blur passes into `encoder`. `radius` is clamped so the
    /// loop cost stays bounded. If `timing_qs` is `Some`, begin/end
    /// timestamps are stamped at the start of the horizontal pass and
    /// the end of the vertical pass.
    pub fn run(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        radius: u32,
        timing_qs: Option<(&wgpu::QuerySet, u32, u32)>,
    ) {
        let r = radius.clamp(1, 32);
        queue.write_buffer(
            &self.params_h,
            0,
            bytemuck::bytes_of(&BlurParams {
                direction: [1, 0],
                radius: r,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &self.params_v,
            0,
            bytemuck::bytes_of(&BlurParams {
                direction: [0, 1],
                radius: r,
                _pad: 0,
            }),
        );

        let gx = self.width.div_ceil(8);
        let gy = self.height.div_ceil(8);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frostify.blur horizontal"),
                timestamp_writes: timing_qs.map(|(qs, begin, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(begin),
                        end_of_pass_write_index: None,
                    }
                }),
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bg_h, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frostify.blur vertical"),
                timestamp_writes: timing_qs.map(|(qs, _, end)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(end),
                    }
                }),
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bg_v, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
    }
}

fn make_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BACKDROP_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
    params: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    })
}
