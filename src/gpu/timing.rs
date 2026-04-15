//! GPU timestamp query resources.
//!
//! Two-slot query set: index 0 = beginning of the opaque pass, index 1 =
//! end of the final pass. The difference is the total wall-clock GPU time
//! spent on the frame's passes. Stage-1 skips per-pass breakdown; M9 can
//! grow this to a 6-slot set if we want opaque / blur / final splits.

/// Must be at least 16 (two u64 results) and aligned to
/// `wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT` (256) for the destination
/// offset. We use a 256-byte resolve + 256-byte readback so the sizes
/// match and readback slicing stays easy.
const RESOLVE_BYTES: wgpu::BufferAddress = 256;

#[derive(Copy, Clone, Debug, Default)]
pub struct FrameTiming {
    pub total_ms: f32,
}

pub struct Timing {
    pub query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
}

impl Timing {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frostify.timing qs"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.timing resolve"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.timing readback"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period_ns = queue.get_timestamp_period();
        Self {
            query_set,
            resolve,
            readback,
            period_ns,
        }
    }

    /// Resolve the two timestamp queries into the readback buffer. Must be
    /// called after both passes have been encoded, before `encoder.finish`.
    pub fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 16);
    }

    /// Blocks on the GPU until the last submitted frame completes, then
    /// reads the two timestamps out and converts to wall-clock ms.
    pub fn read_last(&self, device: &wgpu::Device) -> Option<FrameTiming> {
        let slice = self.readback.slice(0..16);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        rx.recv().ok()?.ok()?;
        let ticks = {
            let view = slice.get_mapped_range();
            let raw: &[u64] = bytemuck::cast_slice(&view);
            [raw[0], raw[1]]
        };
        self.readback.unmap();
        let diff_ticks = ticks[1].saturating_sub(ticks[0]);
        let ms = (diff_ticks as f32) * self.period_ns / 1_000_000.0;
        Some(FrameTiming { total_ms: ms })
    }
}

/// Aggregate frame stats published by the renderer each frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct FrameStats {
    pub cpu_ms: f32,
    pub gpu_ms: f32,
    pub instance_count: u32,
    pub opaque_count: u32,
    pub glass_count: u32,
    pub drawcalls: u32,
    pub dirty_mask: u32,
}

