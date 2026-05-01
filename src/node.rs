//! Retained node tree.
//!
//! Generational-index arena. Nodes carry a [`LayoutStyle`] declaring
//! their sizing/alignment intent; the [`crate::layout::compute_layout`]
//! pass resolves them into absolute [`Node::rect`]s before each flush.
//! `NodeId`s are stable across mutations of *other* nodes — they only
//! invalidate when the specific slot they refer to is reused.

use crate::gpu::{ImageHandle, ShapeInstance, SHAPE_KIND_GLASS, SHAPE_KIND_IMAGE, SHAPE_KIND_RECT};
use crate::layout::{Align, Axis, Justify, Len, LayoutStyle};
use crate::signal::Signal;

/// Tree-level dirty flags.
pub mod dirty {
    pub const NONE: u32 = 0;
    /// Color, opacity, border or shadow style changed.
    pub const VISUAL: u32 = 1 << 0;
    /// Layout style (size, position, padding, gap, justify, align, abs)
    /// changed — requires a layout-pass re-run.
    pub const TRANSFORM: u32 = 1 << 1;
    /// Tree topology changed (add, remove, visibility flip).
    pub const TREE: u32 = 1 << 2;
    /// Glass region or the opaque content under it changed → re-run blur.
    pub const BACKDROP: u32 = 1 << 3;
    pub const ANY: u32 = VISUAL | TRANSFORM | TREE | BACKDROP;
}

/// One text node discovered during flatten. Carries the post-layout
/// absolute position so the caller (GpuContext) can shape + rasterize
/// + append glyph instances without re-walking the tree.
#[derive(Clone, Debug)]
pub struct TextRef {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub opacity: f32,
    pub content: String,
    pub font_size: f32,
    pub line_height: f32,
}

/// One image node discovered during flatten. The atlas lookup happens
/// caller-side (`gpu.build_image_instances`) so the tree stays
/// gpu-free.
#[derive(Clone, Debug)]
pub struct ImageRef {
    pub position: [f32; 2],
    pub size: [f32; 2],
    /// Tint multiplier; `[1,1,1,1]` leaves the image unmodified.
    pub color: [f32; 4],
    pub opacity: f32,
    pub border_radius: [f32; 4],
    pub handle: ImageHandle,
}

/// One interactive rect in the hit-test cache. Produced by
/// `NodeTree::flatten_with_hits` in **topmost-first** order (last-painted
/// first) so hit-test can walk linearly and stop at the first containing
/// rect.
#[derive(Clone, Debug)]
pub struct HitEntry {
    pub node_id: NodeId,
    /// Absolute pixel AABB: `[min_x, min_y, max_x, max_y]`.
    pub bounds: [f32; 4],
}

impl HitEntry {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.bounds[0] && x < self.bounds[2] && y >= self.bounds[1] && y < self.bounds[3]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Rect,
    Glass,
    Text,
    Image,
}

impl ShapeKind {
    pub fn as_u32(self) -> u32 {
        match self {
            ShapeKind::Rect => SHAPE_KIND_RECT,
            ShapeKind::Glass => SHAPE_KIND_GLASS,
            ShapeKind::Text => SHAPE_KIND_RECT,
            ShapeKind::Image => SHAPE_KIND_IMAGE,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

#[derive(Clone, Debug)]
pub struct ShapeStyle {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub border_width: f32,
    pub border_radius: [f32; 4],
    pub shadow_color: [f32; 4],
    pub shadow_offset: [f32; 2],
    pub shadow_blur: f32,
    pub shadow_opacity: f32,
    pub opacity: f32,
    pub kind: ShapeKind,
    /// Glass-only. Backdrop blur radius in logical px. 0 = sharp pass-
    /// through; ~16 = soft frosted look. Scaled to physical px before
    /// reaching the GPU.
    pub blur_amount: f32,
    /// Glass-only. Edge refraction strength in logical px. The SDF
    /// gradient bends backdrop sample UVs outward at the panel rim,
    /// mimicking how a thick glass slab refracts light. 0 disables.
    pub refraction: f32,
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            // Transparent by default: a container node with no explicit
            // color should not render a filled rect. Callers opt in via
            // `.rgba(...)` / `.color(...)`.
            color: [0.0; 4],
            border_color: [0.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            border_radius: [0.0; 4],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            shadow_offset: [0.0; 2],
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
            kind: ShapeKind::Rect,
            blur_amount: 12.0,
            refraction: 0.0,
        }
    }
}

/// System window action bound to a node. When the user left-presses a
/// node tagged with one of these the app shell calls into winit
/// directly (drag the window, exit, minimize, toggle maximize) instead
/// of running normal hit-test press bookkeeping. The node's
/// `NodeInteract` signals (if any) still receive hover updates so the
/// visual can react.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowAction {
    /// Initiate a system window drag (frameless title-bar behaviour).
    DragMove,
    /// Exit the event loop.
    Close,
    /// Minimise the window.
    Minimize,
    /// Toggle the window's maximised state.
    ToggleMaximize,
}

#[derive(Clone, Debug, Default)]
pub struct NodeInteract {
    pub hover: Option<Signal<bool>>,
    pub pressed: Option<Signal<bool>>,
    pub focused: Option<Signal<bool>>,
}

impl NodeInteract {
    pub fn is_any(&self) -> bool {
        self.hover.is_some() || self.pressed.is_some() || self.focused.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct NodeText {
    pub content: String,
    pub font_size: f32,
    pub line_height: f32,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub style: ShapeStyle,
    pub layout: LayoutStyle,
    /// Post-layout absolute rect `[x, y, w, h]`. Written by
    /// [`crate::layout::compute_layout`]; read by `flatten_with_text`.
    pub rect: [f32; 4],
    pub visible: bool,
    pub children: Vec<NodeId>,
    pub interact: NodeInteract,
    pub text: Option<NodeText>,
    pub image: Option<ImageHandle>,
    pub window_action: Option<WindowAction>,
}

impl Node {
    pub fn rect() -> NodeBuilder {
        NodeBuilder {
            node: Node {
                style: ShapeStyle::default(),
                layout: LayoutStyle::default(),
                rect: [0.0; 4],
                visible: true,
                children: Vec::new(),
                interact: NodeInteract::default(),
                text: None,
                image: None,
                window_action: None,
            },
        }
    }

    /// Frosted glass rect. Samples the blurred backdrop behind it.
    pub fn glass() -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Glass;
        b.node.style.color = [1.0, 1.0, 1.0, 0.08];
        b
    }

    /// Text node. Content defaults to `Len::Auto` sizing; layout pass
    /// measures shaped width + `line_height` via the app's measurer.
    pub fn text(content: impl Into<String>, font_size: f32) -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Text;
        b.node.text = Some(NodeText {
            content: content.into(),
            font_size,
            line_height: font_size * 1.25,
        });
        b
    }

    /// Image node sourced from a previously-uploaded atlas handle. Tint
    /// via [`NodeBuilder::color`] / `.rgba()` (default `[1,1,1,1]` =
    /// unmodified). Sized like any other node — `.size_px(w,h)` for
    /// fixed pixels, layout drives Fill/Auto.
    pub fn image(handle: ImageHandle) -> NodeBuilder {
        let mut b = Self::rect();
        b.node.style.kind = ShapeKind::Image;
        b.node.style.color = [1.0, 1.0, 1.0, 1.0];
        b.node.image = Some(handle);
        b
    }
}

pub struct NodeBuilder {
    node: Node,
}

impl NodeBuilder {
    // --- layout ---
    pub fn layout_axis(mut self, a: Axis) -> Self {
        self.node.layout.axis = a;
        self
    }
    pub fn layout_width(mut self, w: Len) -> Self {
        self.node.layout.width = w;
        self
    }
    pub fn layout_height(mut self, h: Len) -> Self {
        self.node.layout.height = h;
        self
    }
    pub fn layout_size(mut self, w: Len, h: Len) -> Self {
        self.node.layout.width = w;
        self.node.layout.height = h;
        self
    }
    pub fn layout_padding(mut self, p: [f32; 4]) -> Self {
        self.node.layout.padding = p;
        self
    }
    pub fn layout_gap(mut self, g: f32) -> Self {
        self.node.layout.gap = g;
        self
    }
    pub fn layout_justify(mut self, j: Justify) -> Self {
        self.node.layout.justify = j;
        self
    }
    pub fn layout_align(mut self, a: Align) -> Self {
        self.node.layout.align = a;
        self
    }
    pub fn layout_abs(mut self, x: f32, y: f32) -> Self {
        self.node.layout.abs = Some([x, y]);
        self
    }

    // --- style ---
    pub fn color(mut self, rgba: [f32; 4]) -> Self {
        self.node.style.color = rgba;
        self
    }
    pub fn rgb(self, r: f32, g: f32, b: f32) -> Self {
        self.color([r, g, b, 1.0])
    }
    pub fn rgba(self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color([r, g, b, a])
    }
    pub fn radius(mut self, r: f32) -> Self {
        self.node.style.border_radius = [r; 4];
        self
    }
    pub fn radii(mut self, tl: f32, tr: f32, bl: f32, br: f32) -> Self {
        self.node.style.border_radius = [tl, tr, bl, br];
        self
    }
    pub fn border(mut self, width: f32, color: [f32; 4]) -> Self {
        self.node.style.border_width = width;
        self.node.style.border_color = color;
        self
    }
    pub fn shadow(mut self, offset: [f32; 2], blur: f32, color: [f32; 4], opacity: f32) -> Self {
        self.node.style.shadow_offset = offset;
        self.node.style.shadow_blur = blur;
        self.node.style.shadow_color = color;
        self.node.style.shadow_opacity = opacity;
        self
    }
    pub fn opacity(mut self, o: f32) -> Self {
        self.node.style.opacity = o;
        self
    }
    pub fn hidden(mut self) -> Self {
        self.node.visible = false;
        self
    }
    pub fn kind(mut self, kind: ShapeKind) -> Self {
        self.node.style.kind = kind;
        self
    }
    /// Per-glass backdrop blur radius (logical px). Typical UI values
    /// 8..32. 0 = no blur (sharp see-through).
    pub fn blur(mut self, px: f32) -> Self {
        self.node.style.blur_amount = px;
        self
    }
    /// Per-glass edge refraction strength (logical px). The backdrop
    /// sample UV is pushed outward by the SDF normal, falling off from
    /// rim to centre. Typical values 4..20. 0 disables.
    pub fn refraction(mut self, px: f32) -> Self {
        self.node.style.refraction = px;
        self
    }
    pub fn line_height(mut self, h: f32) -> Self {
        if let Some(t) = self.node.text.as_mut() {
            t.line_height = h;
        }
        self
    }
    pub fn on_hover(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.hover = Some(signal);
        self
    }
    pub fn on_press(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.pressed = Some(signal);
        self
    }
    pub fn on_focus(mut self, signal: Signal<bool>) -> Self {
        self.node.interact.focused = Some(signal);
        self
    }
    pub fn window_action(mut self, action: WindowAction) -> Self {
        self.node.window_action = Some(action);
        self
    }
    pub fn build(self) -> Node {
        self.node
    }
}

struct Slot {
    generation: u32,
    payload: Option<Node>,
}

#[derive(Default)]
pub struct NodeTree {
    slots: Vec<Slot>,
    free: Vec<u32>,
    roots: Vec<NodeId>,
    dirty: u32,
    /// Count of currently-inserted Glass-kind nodes. Used by mutators
    /// to skip the BACKDROP dirty flag when the tree has no glass —
    /// nothing samples the blurred backdrop in that case, so re-running
    /// the blur pass would be wasted work.
    glass_count: u32,
}

impl NodeTree {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, node: Node) -> NodeId {
        let is_glass = matches!(node.style.kind, ShapeKind::Glass);
        let id = if let Some(idx) = self.free.pop() {
            let slot = &mut self.slots[idx as usize];
            slot.payload = Some(node);
            NodeId {
                index: idx,
                generation: slot.generation,
            }
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                payload: Some(node),
            });
            NodeId {
                index: idx,
                generation: 0,
            }
        };
        if is_glass {
            self.glass_count += 1;
        }
        id
    }

    pub fn add_root(&mut self, node: Node) -> NodeId {
        let id = self.insert(node);
        self.roots.push(id);
        self.dirty |= dirty::TREE;
        id
    }

    pub fn add_child(&mut self, parent: NodeId, node: Node) -> NodeId {
        let id = self.insert(node);
        if let Some(p) = self.get_mut_raw(parent) {
            p.children.push(id);
        }
        self.dirty |= dirty::TREE;
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        let Some(slot) = self.slots.get_mut(id.index as usize) else {
            return;
        };
        if slot.generation != id.generation {
            return;
        }
        let was_glass = slot
            .payload
            .as_ref()
            .map(|n| matches!(n.style.kind, ShapeKind::Glass))
            .unwrap_or(false);
        slot.generation = slot.generation.wrapping_add(1);
        slot.payload = None;
        self.free.push(id.index);
        self.roots.retain(|r| *r != id);
        self.dirty |= dirty::TREE;
        if was_glass {
            self.glass_count = self.glass_count.saturating_sub(1);
            // Removed glass → backdrop no longer needed for it but
            // any remaining glass still samples the same texture; safe
            // to skip BACKDROP. TREE flag drives a full re-flatten which
            // already triggers a re-blur via set_instances if needed.
        }
    }

    // --- layout-mutating setters (flag TRANSFORM + BACKDROP conservatively) ---

    pub fn set_layout_width(&mut self, id: NodeId, w: Len) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.width != w {
                n.layout.width = w;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_height(&mut self, id: NodeId, h: Len) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.height != h {
                n.layout.height = h;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_abs(&mut self, id: NodeId, pos: Option<[f32; 2]>) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.abs != pos {
                n.layout.abs = pos;
                self.dirty |= mask;
            }
    }

    /// Convenience for animated position binds: forces `layout.abs =
    /// Some([x,y])`. Skips the dirty flag bump if the value didn't move.
    pub fn set_layout_pos_abs(&mut self, id: NodeId, pos: [f32; 2]) {
        self.set_layout_abs(id, Some(pos));
    }

    /// Convenience for animated size binds: forces both axes to `Px`.
    pub fn set_layout_size_px(&mut self, id: NodeId, size: [f32; 2]) {
        let w = Len::Px(size[0]);
        let h = Len::Px(size[1]);
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id) {
            let changed = n.layout.width != w || n.layout.height != h;
            if changed {
                n.layout.width = w;
                n.layout.height = h;
                self.dirty |= mask;
            }
        }
    }

    pub fn set_layout_padding(&mut self, id: NodeId, padding: [f32; 4]) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.padding != padding {
                n.layout.padding = padding;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_gap(&mut self, id: NodeId, gap: f32) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.gap != gap {
                n.layout.gap = gap;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_justify(&mut self, id: NodeId, j: Justify) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.justify != j {
                n.layout.justify = j;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_align(&mut self, id: NodeId, a: Align) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.align != a {
                n.layout.align = a;
                self.dirty |= mask;
            }
    }

    pub fn set_layout_axis(&mut self, id: NodeId, ax: Axis) {
        let mask = self.transform_mask();
        if let Some(n) = self.get_mut_raw(id)
            && n.layout.axis != ax {
                n.layout.axis = ax;
                self.dirty |= mask;
            }
    }

    pub fn set_color(&mut self, id: NodeId, color: [f32; 4]) {
        let has_glass = self.has_glass();
        if let Some(n) = self.get_mut_raw(id)
            && n.style.color != color {
                // Glass + Image render in the final pass only, so they
                // don't enter the blurred backdrop. And without any
                // glass node sampling it, the blur pass is skipped
                // anyway — no need to flag BACKDROP.
                let is_opaque_change =
                    !matches!(n.style.kind, ShapeKind::Glass | ShapeKind::Image);
                n.style.color = color;
                self.dirty |= dirty::VISUAL;
                if is_opaque_change && has_glass {
                    self.dirty |= dirty::BACKDROP;
                }
            }
    }

    pub fn set_opacity(&mut self, id: NodeId, opacity: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && n.style.opacity != opacity {
                n.style.opacity = opacity;
                self.dirty |= dirty::VISUAL;
            }
    }

    pub fn set_text(&mut self, id: NodeId, content: impl Into<String>) {
        let content = content.into();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(t) = n.text.as_mut()
                && t.content != content {
                    t.content = content;
                    // Text width changes → relayout (Auto-sized text).
                    self.dirty |= dirty::VISUAL | dirty::TRANSFORM;
                }
    }

    pub fn set_font_size(&mut self, id: NodeId, font_size: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(t) = n.text.as_mut()
                && t.font_size != font_size {
                    let old_ratio = t.line_height / t.font_size.max(0.0001);
                    t.font_size = font_size;
                    t.line_height = font_size * old_ratio;
                    self.dirty |= dirty::VISUAL | dirty::TRANSFORM;
                }
    }

    pub fn set_visible(&mut self, id: NodeId, visible: bool) {
        if let Some(n) = self.get_mut_raw(id)
            && n.visible != visible {
                n.visible = visible;
                self.dirty |= dirty::TREE;
            }
    }

    pub fn dirty(&self) -> u32 {
        self.dirty
    }

    /// True when at least one Glass-kind node lives in the tree. Used
    /// to gate the BACKDROP dirty flag on layout/visual mutations —
    /// without glass, nothing samples the blurred backdrop so re-running
    /// the blur is wasted work.
    pub fn has_glass(&self) -> bool {
        self.glass_count > 0
    }

    /// Mask to OR into `self.dirty` for any layout-mutating setter.
    /// Drops BACKDROP when the tree has no glass.
    fn transform_mask(&self) -> u32 {
        if self.has_glass() {
            dirty::TRANSFORM | dirty::BACKDROP
        } else {
            dirty::TRANSFORM
        }
    }

    pub fn take_dirty(&mut self) -> u32 {
        let d = self.dirty;
        self.dirty = dirty::NONE;
        d
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty |= dirty::ANY;
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.payload.as_ref()
    }

    pub fn get_mut_raw(&mut self, id: NodeId) -> Option<&mut Node> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.payload.as_mut()
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.payload.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    /// DFS preorder flatten into a single ordered event stream,
    /// reading post-layout `Node.rect`. Parent opacity multiplies
    /// down. Painter's order across all kinds — caller resolves
    /// Text/Image events into GPU instances at their event index so
    /// layering is preserved. Hit cache is topmost-first.
    pub fn flatten(&self) -> (Vec<FlatEvent>, Vec<HitEntry>) {
        let mut events = Vec::with_capacity(self.len());
        let mut hits = Vec::new();
        for root in &self.roots {
            self.flatten_into(*root, 1.0, &mut events, &mut hits);
        }
        hits.reverse();
        (events, hits)
    }

    #[cfg(test)]
    fn dirty_for_test(&self) -> u32 {
        self.dirty
    }

    fn flatten_into(
        &self,
        id: NodeId,
        parent_opacity: f32,
        events: &mut Vec<FlatEvent>,
        hits: &mut Vec<HitEntry>,
    ) {
        let Some(node) = self.get(id) else { return };
        if !node.visible {
            return;
        }
        let rect = node.rect;
        let abs = [rect[0], rect[1]];
        let size = [rect[2], rect[3]];
        let opacity = parent_opacity * node.style.opacity;
        match node.style.kind {
            ShapeKind::Rect | ShapeKind::Glass => {
                // For glass, repurpose backdrop_uv_rect.xy to carry bevel
                // params (the field is ignored by the glass branch's UV
                // sampling since glass uses screen-space UVs).
                let extras = if matches!(node.style.kind, ShapeKind::Glass) {
                    [
                        node.style.blur_amount,
                        node.style.refraction,
                        0.0,
                        0.0,
                    ]
                } else {
                    [0.0; 4]
                };
                events.push(FlatEvent::Shape(ShapeInstance {
                    color: node.style.color,
                    border_color: node.style.border_color,
                    shadow_color: node.style.shadow_color,
                    border_radius: node.style.border_radius,
                    backdrop_uv_rect: extras,
                    position: abs,
                    size,
                    shadow_offset: node.style.shadow_offset,
                    shape_kind: node.style.kind.as_u32(),
                    _pad0: 0.0,
                    border_width: node.style.border_width,
                    shadow_blur: node.style.shadow_blur,
                    shadow_opacity: node.style.shadow_opacity,
                    opacity,
                }));
            }
            ShapeKind::Text => {
                if let Some(t) = node.text.as_ref() {
                    events.push(FlatEvent::Text(TextRef {
                        position: abs,
                        color: node.style.color,
                        opacity,
                        content: t.content.clone(),
                        font_size: t.font_size,
                        line_height: t.line_height,
                    }));
                }
            }
            ShapeKind::Image => {
                if let Some(handle) = node.image {
                    events.push(FlatEvent::Image(ImageRef {
                        position: abs,
                        size,
                        color: node.style.color,
                        opacity,
                        border_radius: node.style.border_radius,
                        handle,
                    }));
                }
            }
        }
        if node.interact.is_any() || node.window_action.is_some() {
            hits.push(HitEntry {
                node_id: id,
                bounds: [abs[0], abs[1], abs[0] + size[0], abs[1] + size[1]],
            });
        }
        for &child in &node.children {
            self.flatten_into(child, opacity, events, hits);
        }
    }
}

/// A single node's contribution to the rendered frame, in declared
/// order. Text/Image still need atlas resolution before they become
/// GPU instances; the caller walks the vec in order so layering is
/// preserved across all kinds.
#[derive(Clone, Debug)]
pub enum FlatEvent {
    Shape(ShapeInstance),
    Text(TextRef),
    Image(ImageRef),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Len;

    #[test]
    fn glass_count_tracks_inserts_and_removes() {
        let mut t = NodeTree::new();
        assert!(!t.has_glass());
        let a = t.add_root(Node::rect().build());
        assert!(!t.has_glass());
        let g = t.add_root(Node::glass().build());
        assert!(t.has_glass());
        let g2 = t.add_root(Node::glass().build());
        assert!(t.has_glass());
        t.remove(g);
        assert!(t.has_glass());
        t.remove(g2);
        assert!(!t.has_glass());
        t.remove(a);
        assert!(!t.has_glass());
    }

    #[test]
    fn layout_setter_skips_backdrop_without_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        t.take_dirty();
        t.set_layout_width(id, Len::Px(50.0));
        let d = t.dirty_for_test();
        assert!(d & dirty::TRANSFORM != 0);
        assert!(d & dirty::BACKDROP == 0, "no glass → no BACKDROP flag");
    }

    #[test]
    fn layout_setter_flags_backdrop_with_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        let _g = t.add_root(Node::glass().build());
        t.take_dirty();
        t.set_layout_width(id, Len::Px(50.0));
        let d = t.dirty_for_test();
        assert!(d & dirty::TRANSFORM != 0);
        assert!(d & dirty::BACKDROP != 0);
    }

    #[test]
    fn set_color_skips_backdrop_without_glass() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        t.take_dirty();
        t.set_color(id, [0.5, 0.5, 0.5, 1.0]);
        let d = t.dirty_for_test();
        assert!(d & dirty::VISUAL != 0);
        assert!(d & dirty::BACKDROP == 0);
    }

    #[test]
    fn set_color_on_glass_never_flags_backdrop() {
        let mut t = NodeTree::new();
        let g = t.add_root(Node::glass().build());
        t.take_dirty();
        t.set_color(g, [1.0, 0.0, 0.0, 0.5]);
        let d = t.dirty_for_test();
        assert!(d & dirty::VISUAL != 0);
        assert!(
            d & dirty::BACKDROP == 0,
            "glass color change doesn't enter the backdrop"
        );
    }
}
