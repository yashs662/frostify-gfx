use std::sync::Arc;

use winit::window::Window;

use super::blur::BlurResources;
use super::instance::{FrameUniform, ShapeInstance};
use super::overdraw::OverdrawResources;
use super::pipeline::ShapePipeline;
use super::timing::{FrameTiming, Timing};

/// Owns every wgpu handle the renderer touches.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,

    pub shape: ShapePipeline,
    pub blur: BlurResources,
    pub overdraw: OverdrawResources,
    overdraw_mode: bool,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    shape_bg: wgpu::BindGroup,
    glass_bg: wgpu::BindGroup,

    /// Secondary instance buffer for debug overlays (HUD bar gauges,
    /// future inspector outlines). Drawn at the end of the final pass
    /// over the regular scene. Has no effect on the backdrop pass.
    overlay_buffer: wgpu::Buffer,
    overlay_capacity: u64,
    overlay_bg: wgpu::BindGroup,
    overlay_count: u32,

    instance_count: u32,
    opaque_count: u32,
    /// Needs a re-run of the blur compute pass on the next render. Set by
    /// `set_instances`; cleared after render.
    backdrop_dirty: bool,

    /// Timestamp query resources. `Some` when the adapter advertises
    /// `Features::TIMESTAMP_QUERY`, `None` otherwise. Reads happen on
    /// demand via `take_last_timing`.
    timing: Option<Timing>,
    /// Render-pass + compute drawcall counter for the most recent frame.
    last_drawcalls: u32,
    /// Cached frame timing read at the end of the last render. `None`
    /// when timing isn't available or hasn't been read yet.
    last_timing: Option<FrameTiming>,
}

impl GpuContext {
    pub fn new(window: Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("no suitable adapter");

        let adapter_features = adapter.features();
        let want_timing = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut required_features = wgpu::Features::empty();
        if want_timing {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frostify-gfx device"),
                required_features,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device request failed");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                *f == wgpu::TextureFormat::Rgba8UnormSrgb
                    || *f == wgpu::TextureFormat::Bgra8UnormSrgb
            })
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::PreMultiplied)
            .unwrap_or(caps.alpha_modes[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        log::info!(
            "gpu init: format={format:?} alpha={alpha_mode:?} size={width}x{height}"
        );

        let shape = ShapePipeline::new(&device, format);
        let blur = BlurResources::new(&device, width, height);
        let overdraw =
            OverdrawResources::new(&device, width, height, format, &shape.shape_bgl);
        let timing = if want_timing {
            Some(Timing::new(&device, &queue))
        } else {
            None
        };
        log::info!("gpu timing: {}", if timing.is_some() { "on" } else { "off" });

        // Allocate an initial instance buffer with room for one shape.
        let instance_capacity: u64 = 16;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.instance ssbo"),
            size: instance_capacity * std::mem::size_of::<ShapeInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shape_bg = make_shape_bg(&device, &shape, &instance_buffer);
        let glass_bg = make_glass_bg(&device, &shape, &blur);

        let overlay_capacity: u64 = 16;
        let overlay_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.overlay ssbo"),
            size: overlay_capacity * std::mem::size_of::<ShapeInstance>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_bg = make_shape_bg(&device, &shape, &overlay_buffer);

        // Write initial frame uniform.
        queue.write_buffer(
            &shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [width as f32, height as f32],
                _pad: [0.0; 2],
            }),
        );

        Self {
            instance,
            device,
            queue,
            surface,
            surface_config,
            window,
            shape,
            blur,
            overdraw,
            overdraw_mode: false,
            instance_buffer,
            instance_capacity,
            shape_bg,
            glass_bg,
            overlay_buffer,
            overlay_capacity,
            overlay_bg,
            overlay_count: 0,
            instance_count: 0,
            opaque_count: 0,
            backdrop_dirty: true,
            timing,
            last_drawcalls: 0,
            last_timing: None,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.queue.write_buffer(
            &self.shape.frame_buffer,
            0,
            bytemuck::bytes_of(&FrameUniform {
                screen_size: [
                    self.surface_config.width as f32,
                    self.surface_config.height as f32,
                ],
                _pad: [0.0; 2],
            }),
        );
        self.blur
            .resize(&self.device, self.surface_config.width, self.surface_config.height);
        // Blurred view changed — rebuild the glass bind group.
        self.glass_bg = make_glass_bg(&self.device, &self.shape, &self.blur);
        self.overdraw.resize(
            &self.device,
            self.surface_config.width,
            self.surface_config.height,
        );
        self.backdrop_dirty = true;
    }

    pub fn overdraw_mode(&self) -> bool {
        self.overdraw_mode
    }

    pub fn set_overdraw(&mut self, on: bool) {
        self.overdraw_mode = on;
    }

    /// Upload a complete instance list. Partitioned caller-side: the first
    /// `opaque_count` entries are opaque shapes (drawn to backdrop + final),
    /// the remainder are glass shapes (drawn only in the final pass).
    pub fn set_instances(&mut self, instances: &[ShapeInstance], opaque_count: u32) {
        let needed = instances.len() as u64;
        if needed > self.instance_capacity {
            let mut new_cap = self.instance_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frostify.instance ssbo"),
                size: new_cap * std::mem::size_of::<ShapeInstance>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
            self.shape_bg = make_shape_bg(&self.device, &self.shape, &self.instance_buffer);
        }

        if !instances.is_empty() {
            self.queue
                .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
        self.instance_count = instances.len() as u32;
        self.opaque_count = opaque_count.min(self.instance_count);
        // Instance content changed → backdrop must be rebuilt.
        self.backdrop_dirty = true;
    }

    pub fn glass_count(&self) -> u32 {
        self.instance_count - self.opaque_count
    }

    /// Upload a list of overlay instances drawn after the main scene.
    /// Pass an empty slice to clear. Same growth scheme as the main
    /// instance buffer.
    pub fn set_overlay_instances(&mut self, instances: &[ShapeInstance]) {
        let needed = instances.len() as u64;
        if needed > self.overlay_capacity {
            let mut new_cap = self.overlay_capacity.max(1);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.overlay_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frostify.overlay ssbo"),
                size: new_cap * std::mem::size_of::<ShapeInstance>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.overlay_capacity = new_cap;
            self.overlay_bg = make_shape_bg(&self.device, &self.shape, &self.overlay_buffer);
        }
        if !instances.is_empty() {
            self.queue
                .write_buffer(&self.overlay_buffer, 0, bytemuck::cast_slice(instances));
        }
        self.overlay_count = instances.len() as u32;
    }

    pub fn mark_backdrop_dirty(&mut self) {
        self.backdrop_dirty = true;
    }

    /// Encode the opaque pass, blur pass (if needed), and final pass into
    /// `encoder`. `final_view` is the render target for the surface pass.
    fn encode_frame(&mut self, encoder: &mut wgpu::CommandEncoder, final_view: &wgpu::TextureView) {
        let mut drawcalls: u32 = 0;
        let timing_qs = self.timing.as_ref().map(|t| &t.query_set);

        // ---- Pass A: opaque shapes → backdrop_tex ------------------------
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.backdrop pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur.backdrop_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.opaque_count > 0 {
                rpass.set_pipeline(&self.shape.opaque_pipeline);
                rpass.set_bind_group(0, &self.shape_bg, &[]);
                rpass.draw(0..6, 0..self.opaque_count);
                drawcalls += 1;
            }
        }

        // ---- Pass B: separable Gaussian blur ----------------------------
        // Always runs when glass is present. Skipped if no glass in scene.
        let has_glass = self.glass_count() > 0;
        if has_glass {
            // Radius is fixed stage-1; M9 will key it off per-shape roughness.
            let radius: u32 = 16;
            self.blur.run(&self.queue, encoder, radius);
            self.backdrop_dirty = false;
            drawcalls += 2; // two separable compute dispatches
        }

        // ---- Pass C: final surface ------------------------------------
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frostify.final pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timing_qs.map(|qs| wgpu::RenderPassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(1),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count > 0 {
                rpass.set_pipeline(&self.shape.final_pipeline);
                rpass.set_bind_group(0, &self.shape_bg, &[]);
                rpass.set_bind_group(1, &self.glass_bg, &[]);
                rpass.draw(0..6, 0..self.instance_count);
                drawcalls += 1;
            }
            if self.overlay_count > 0 {
                rpass.set_pipeline(&self.shape.final_pipeline);
                rpass.set_bind_group(0, &self.overlay_bg, &[]);
                rpass.set_bind_group(1, &self.glass_bg, &[]);
                rpass.draw(0..6, 0..self.overlay_count);
                drawcalls += 1;
            }
        }

        // ---- Pass D (optional): overdraw count + compose --------------
        // When toggled on, count shape coverage into an Rgba16Float
        // accumulator, then re-render the swapchain with a heatmap of the
        // count. The final pass already cleared and drew the scene; the
        // compose pass overwrites it with the heatmap (LoadOp::Clear).
        if self.overdraw_mode {
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("frostify.overdraw count"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.overdraw.count_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if self.instance_count > 0 {
                    rpass.set_pipeline(&self.overdraw.count_pipeline);
                    rpass.set_bind_group(0, &self.shape_bg, &[]);
                    rpass.draw(0..6, 0..self.instance_count);
                    drawcalls += 1;
                }
            }
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("frostify.overdraw compose"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: final_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.overdraw.compose_pipeline);
                rpass.set_bind_group(0, &self.overdraw.compose_bg, &[]);
                rpass.draw(0..6, 0..1);
                drawcalls += 1;
            }
        }

        if let Some(t) = self.timing.as_ref() {
            t.encode_resolve(encoder);
        }

        self.last_drawcalls = drawcalls;
    }

    /// Acquire, render, present.
    pub fn render_frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("surface validation error");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frostify-gfx frame"),
            });

        self.encode_frame(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.read_timing_after_submit();
        self.window.pre_present_notify();
        frame.present();
    }

    /// Read the resolved timestamp pair from the last submitted frame.
    /// Stalls the CPU on the GPU. Skipped when timing is unavailable.
    fn read_timing_after_submit(&mut self) {
        let Some(t) = self.timing.as_ref() else {
            self.last_timing = None;
            return;
        };
        self.last_timing = t.read_last(&self.device);
    }

    /// Last-frame stats. Drawcall + timing values come from the encoder /
    /// query readback; instance counts mirror the most recent
    /// `set_instances` call.
    pub fn last_frame_stats(&self) -> super::timing::FrameStats {
        super::timing::FrameStats {
            cpu_ms: 0.0,
            gpu_ms: self.last_timing.map(|t| t.total_ms).unwrap_or(0.0),
            instance_count: self.instance_count,
            opaque_count: self.opaque_count,
            glass_count: self.glass_count(),
            drawcalls: self.last_drawcalls,
            dirty_mask: 0,
        }
    }

    /// True when `Features::TIMESTAMP_QUERY` is active and `last_frame_stats`
    /// will return a meaningful `gpu_ms`.
    pub fn timing_enabled(&self) -> bool {
        self.timing.is_some()
    }

    /// Render one frame into an offscreen RGBA texture and return raw
    /// pixels + dimensions. Used by the F2 screenshot path. Blocks on the
    /// GPU map. Non-hot path.
    pub fn capture_rgba(&mut self) -> (Vec<u8>, u32, u32) {
        let width = self.surface_config.width;
        let height = self.surface_config.height;

        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frostify.capture target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Readback buffer. Row pitch must be 256-aligned (COPY_BYTES_PER_ROW_ALIGNMENT).
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let readback_size = (padded_bpr as u64) * height as u64;

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.capture readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frostify.capture encoder"),
            });

        self.encode_frame(&mut encoder, &view);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        self.read_timing_after_submit();

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv()
            .expect("map channel closed")
            .expect("map failed");

        let view = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded_bpr * height) as usize);
        for row in 0..height {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            out.extend_from_slice(&view[start..end]);
        }
        drop(view);
        readback.unmap();

        if matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in out.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }

        (out, width, height)
    }
}

fn make_shape_bg(
    device: &wgpu::Device,
    shape: &ShapePipeline,
    instance_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frostify.shape bg"),
        layout: &shape.shape_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: shape.frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instance_buffer.as_entire_binding(),
            },
        ],
    })
}

fn make_glass_bg(
    device: &wgpu::Device,
    shape: &ShapePipeline,
    blur: &BlurResources,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frostify.glass bg"),
        layout: &shape.glass_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&blur.blurred_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&blur.sampler),
            },
        ],
    })
}
