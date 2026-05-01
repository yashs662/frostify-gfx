//! Text shaping + glyph rasterization.
//!
//! Thin wrapper around cosmic-text's `FontSystem` + `SwashCache`.
//! Produces [`ShapedGlyph`]s (position + `CacheKey`) and on-demand
//! R8 masks via [`TextResources::rasterize`]. The atlas that holds
//! those masks and the draw-instance emission both live outside this
//! module — see [`crate::gpu`] for the atlas and
//! [`crate::node::ShapeKind::Text`] for the scene-graph hookup.
//!
//! Stage 1 uses system fonts (SansSerif family, sans bundled assets)
//! and rejects color-emoji glyphs so the atlas can stay R8Unorm.

use cosmic_text::{
    Attrs, Buffer, CacheKey, Family, FontSystem, Metrics, Shaping, SwashCache, SwashContent,
};

/// Owns the shaping engine + rasterized-glyph cache. One per `App`.
pub struct TextResources {
    font_system: FontSystem,
    swash_cache: SwashCache,
}

/// One shaped glyph: an atlas key and its physical-pixel pen position.
/// The final screen position is `pen + (placement.left, -placement.top)`
/// where `placement` comes from [`TextResources::rasterize`].
#[derive(Copy, Clone, Debug)]
pub struct ShapedGlyph {
    pub cache_key: CacheKey,
    pub x: i32,
    pub y: i32,
}

/// Result of rasterizing one glyph to an R8 coverage mask.
#[derive(Debug)]
pub struct RasterizedGlyph {
    pub width: u32,
    pub height: u32,
    /// Horizontal bearing from pen origin to left of bitmap.
    pub left: i32,
    /// Vertical bearing from pen origin to top of bitmap (positive = above baseline).
    pub top: i32,
    /// Row-major R8 coverage, `width * height` bytes.
    pub data: Vec<u8>,
}

/// Bounding box of a shaped run, in physical pixels relative to the
/// pen origin at `(0, 0)` on the first baseline.
#[derive(Copy, Clone, Debug, Default)]
pub struct TextMetrics {
    pub width: f32,
    pub height: f32,
}

impl TextResources {
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
        }
    }

    /// Shape `text` at `size_px` line-height `line_height_px`. Returns
    /// physical glyph positions and cache keys (for atlas lookup).
    pub fn shape(&mut self, text: &str, size_px: f32, line_height_px: f32) -> Vec<ShapedGlyph> {
        let mut buf = Buffer::new(&mut self.font_system, Metrics::new(size_px, line_height_px));
        let attrs = Attrs::new().family(Family::SansSerif);
        buf.set_text(text, &attrs, Shaping::Advanced, None);
        buf.shape_until_scroll(&mut self.font_system, false);
        let mut out = Vec::new();
        for run in buf.layout_runs() {
            for g in run.glyphs.iter() {
                let phys = g.physical((0.0, run.line_y), 1.0);
                out.push(ShapedGlyph {
                    cache_key: phys.cache_key,
                    x: phys.x,
                    y: phys.y,
                });
            }
        }
        out
    }

    /// Shape + measure bounding box in one call.
    pub fn measure(&mut self, text: &str, size_px: f32, line_height_px: f32) -> TextMetrics {
        let mut buf = Buffer::new(&mut self.font_system, Metrics::new(size_px, line_height_px));
        let attrs = Attrs::new().family(Family::SansSerif);
        buf.set_text(text, &attrs, Shaping::Advanced, None);
        buf.shape_until_scroll(&mut self.font_system, false);
        let mut w: f32 = 0.0;
        let mut lines: u32 = 0;
        for run in buf.layout_runs() {
            w = w.max(run.line_w);
            lines += 1;
        }
        TextMetrics {
            width: w,
            height: lines as f32 * line_height_px,
        }
    }

    /// Rasterize one glyph. Returns `None` for color-emoji glyphs
    /// (stage-1 atlas is R8 only) or missing glyphs.
    pub fn rasterize(&mut self, key: CacheKey) -> Option<RasterizedGlyph> {
        let img = self
            .swash_cache
            .get_image(&mut self.font_system, key)
            .as_ref()?;
        if !matches!(img.content, SwashContent::Mask) {
            return None;
        }
        Some(RasterizedGlyph {
            width: img.placement.width,
            height: img.placement.height,
            left: img.placement.left,
            top: img.placement.top,
            data: img.data.clone(),
        })
    }
}

impl Default for TextResources {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::layout::Measurer for TextResources {
    fn measure_text(&mut self, content: &str, font_size: f32, line_height: f32) -> [f32; 2] {
        let m = self.measure(content, font_size, line_height);
        [m.width, m.height]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_produces_glyph_per_ascii_letter() {
        let mut t = TextResources::new();
        let g = t.shape("abc", 16.0, 20.0);
        assert_eq!(g.len(), 3);
        // Each glyph has a distinct cache key (different glyph ids).
        let mut keys: Vec<_> = g.iter().map(|s| s.cache_key).collect();
        keys.sort_by_key(|k| k.glyph_id);
        keys.dedup();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn rasterize_roundtrip_produces_mask_bytes() {
        let mut t = TextResources::new();
        let g = t.shape("A", 32.0, 40.0);
        assert_eq!(g.len(), 1);
        let r = t.rasterize(g[0].cache_key).expect("rasterize A");
        assert!(r.width > 0 && r.height > 0, "empty bitmap");
        assert_eq!(
            r.data.len(),
            (r.width * r.height) as usize,
            "row-major R8 invariant"
        );
        assert!(
            r.data.iter().any(|&b| b > 0),
            "glyph bitmap has no non-zero coverage"
        );
    }

    #[test]
    fn measure_reports_nonzero_box() {
        let mut t = TextResources::new();
        let m = t.measure("hello", 16.0, 20.0);
        assert!(m.width > 0.0);
        assert!(m.height >= 20.0);
    }
}
