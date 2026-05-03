//! Retained node tree.
//!
//! Generational-index arena. Nodes carry a [`LayoutStyle`] declaring
//! their sizing/alignment intent; the [`crate::layout::compute_layout`]
//! pass resolves them into absolute [`Node::rect`]s before each flush.
//! `NodeId`s are stable across mutations of *other* nodes — they only
//! invalidate when the specific slot they refer to is reused.

use crate::gpu::{ImageHandle, NO_CLIP, ShapeInstance, SHAPE_KIND_GLASS, SHAPE_KIND_IMAGE, SHAPE_KIND_RECT};
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
    /// Scroll offset or scrollbar interaction state changed. Triggers a
    /// re-flatten (offset propagates to child positions, bar
    /// hover/active flip thumb color) but **does not** need
    /// `compute_layout` to re-run — node `rect`s are still valid. Kept
    /// separate from `TRANSFORM` so a fast drag doesn't re-shape text
    /// + re-measure flex on every cursor-move event.
    pub const SCROLL: u32 = 1 << 4;
    pub const ANY: u32 = VISUAL | TRANSFORM | TREE | BACKDROP | SCROLL;
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
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    /// `crate::gpu::NO_CLIP` when none. Stamped onto every glyph
    /// instance built from this ref.
    pub clip_rect: [f32; 4],
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
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    pub clip_rect: [f32; 4],
}

/// One interactive rect in the hit-test cache. Produced by
/// `NodeTree::flatten_with_hits` in **topmost-first** order (last-painted
/// first) so hit-test can walk linearly and stop at the first containing
/// rect.
#[derive(Clone, Debug)]
pub struct HitEntry {
    pub node_id: NodeId,
    /// Absolute pixel AABB: `[min_x, min_y, max_x, max_y]`.
    /// Already includes any ancestor scroll offset — screen-space.
    pub bounds: [f32; 4],
    /// Scissor rect propagated from the nearest Scroll/Hidden ancestor.
    /// Cursor outside this rect must miss this entry even if `bounds`
    /// would contain it. `crate::gpu::NO_CLIP` when no ancestor clips.
    pub clip_rect: [f32; 4],
}

impl HitEntry {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if x < self.bounds[0] || x >= self.bounds[2] || y < self.bounds[1] || y >= self.bounds[3] {
            return false;
        }
        x >= self.clip_rect[0]
            && x < self.clip_rect[2]
            && y >= self.clip_rect[1]
            && y < self.clip_rect[3]
    }
}

/// One scrollable container discovered during flatten. Wheel input
/// finds the topmost ScrollHit under the cursor and walks
/// `ancestor_chain` for edge-bubble (innermost-first; self is at index
/// 0). Built only for nodes with `layout.scrolls()`.
#[derive(Clone, Debug)]
pub struct ScrollHit {
    pub node_id: NodeId,
    /// Absolute, post-offset bounds — same convention as `HitEntry`.
    pub bounds: [f32; 4],
    pub clip_rect: [f32; 4],
    /// Innermost-first chain including self at `[0]`. Wheel bubble
    /// walks this on edge consumption: self first, then each scroll
    /// ancestor outward.
    pub ancestor_chain: Vec<NodeId>,
}

impl ScrollHit {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if x < self.bounds[0] || x >= self.bounds[2] || y < self.bounds[1] || y >= self.bounds[3] {
            return false;
        }
        x >= self.clip_rect[0]
            && x < self.clip_rect[2]
            && y >= self.clip_rect[1]
            && y < self.clip_rect[3]
    }
}

/// Which axis a scrollbar belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScrollAxis {
    X,
    Y,
}

impl ScrollAxis {
    pub fn index(self) -> usize {
        match self {
            ScrollAxis::X => 0,
            ScrollAxis::Y => 1,
        }
    }
}

/// Edge a scrollbar attaches to. `End` is the conventional side
/// (right for Y, bottom for X); `Start` flips it (left / top).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BarSide {
    Start,
    End,
}

/// Per-scrollbar visual + behavior config. Lives on `ScrollState`. All
/// pixel fields are in **logical** units — emit scales them to physical
/// at flatten time.
#[derive(Copy, Clone, Debug)]
pub struct ScrollbarStyle {
    pub track_color: [f32; 4],
    pub thumb_color: [f32; 4],
    pub thumb_hover_color: [f32; 4],
    pub thumb_active_color: [f32; 4],
    pub thickness: f32,
    pub min_thumb: f32,
    pub margin: f32,
    pub radius: f32,
    pub y_side: BarSide,
    pub x_side: BarSide,
    pub fade_seconds: f32,
    /// Don't pop the bar when scroll input arrives — only show it when
    /// the pointer enters the bar AABB or while a drag is in flight.
    /// Default false.
    pub auto_hide: bool,
    /// Pin `bar_alpha` to 1 (never fade). Useful for desktop apps
    /// where always-on bars are expected.
    pub always_visible: bool,
}

impl Default for ScrollbarStyle {
    fn default() -> Self {
        Self {
            track_color: [1.0, 1.0, 1.0, 0.10],
            thumb_color: [1.0, 1.0, 1.0, 0.45],
            thumb_hover_color: [1.0, 1.0, 1.0, 0.65],
            thumb_active_color: [1.0, 1.0, 1.0, 0.85],
            thickness: 4.0,
            min_thumb: 24.0,
            margin: 4.0,
            radius: 2.0,
            y_side: BarSide::End,
            x_side: BarSide::End,
            fade_seconds: 0.8,
            auto_hide: false,
            always_visible: false,
        }
    }
}

impl ScrollbarStyle {
    pub fn track_color(mut self, c: [f32; 4]) -> Self { self.track_color = c; self }
    pub fn thumb_color(mut self, c: [f32; 4]) -> Self { self.thumb_color = c; self }
    pub fn thumb_hover_color(mut self, c: [f32; 4]) -> Self { self.thumb_hover_color = c; self }
    pub fn thumb_active_color(mut self, c: [f32; 4]) -> Self { self.thumb_active_color = c; self }
    pub fn thickness(mut self, px: f32) -> Self { self.thickness = px; self }
    pub fn min_thumb(mut self, px: f32) -> Self { self.min_thumb = px; self }
    pub fn margin(mut self, px: f32) -> Self { self.margin = px; self }
    pub fn radius(mut self, px: f32) -> Self { self.radius = px; self }
    pub fn y_side(mut self, side: BarSide) -> Self { self.y_side = side; self }
    pub fn x_side(mut self, side: BarSide) -> Self { self.x_side = side; self }
    pub fn fade(mut self, seconds: f32) -> Self { self.fade_seconds = seconds.max(0.0); self }
    pub fn auto_hide(mut self, on: bool) -> Self { self.auto_hide = on; self }
    pub fn always_visible(mut self, on: bool) -> Self { self.always_visible = on; self }
}

/// One scrollbar AABB pair surfaced from flatten. Drives pointer
/// hover/click/drag routing in `input.rs`. Emitted for every active
/// axis on every visible scroll container, regardless of whether the
/// bar is currently rendered (`bar_alpha == 0`) — input still wants
/// to detect hover-enter on the bar region to bring it back.
#[derive(Clone, Debug)]
pub struct ScrollbarHit {
    pub node_id: NodeId,
    pub axis: ScrollAxis,
    /// Track AABB `[min_x, min_y, max_x, max_y]` in screen space.
    pub track: [f32; 4],
    /// Thumb AABB inside the track at the current scroll position.
    pub thumb: [f32; 4],
    pub clip_rect: [f32; 4],
    /// Maximum scroll offset in logical *physical* px on this axis
    /// (`content - rect`). Cached so input can map track-clicks
    /// directly without a tree lookup.
    pub max_offset: f32,
    /// Track travel = `track_len - thumb_len`. The pixel range a thumb
    /// drag covers; cached for the same reason as `max_offset`.
    pub track_travel: f32,
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
    /// Glass-only. Frosted-texture variation in logical px. Per-fragment
    /// hash scatters the backdrop sample UV by `roughness * pixel_of_mip`
    /// so the surface looks pebbled rather than mirror-smooth. 0
    /// disables; ~1 = subtle frost, ~3 = pronounced.
    pub roughness: f32,
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
            roughness: 0.0,
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

/// Per-node scroll state. Allocated only on containers whose layout has
/// `overflow_x == Scroll || overflow_y == Scroll`. `current` is what the
/// flatten pass reads; `target` is what wheel input pushes. Each tick
/// `current` exponentially eases toward `target`.
#[derive(Copy, Clone, Debug)]
pub struct ScrollState {
    pub current: [f32; 2],
    pub target: [f32; 2],
    /// Exponential ease rate. Higher = snappier. Default 12 ≈ 100 ms
    /// time-to-converge.
    pub stiffness: f32,
    /// When true, `target` is allowed past the content edge; the spring
    /// pulls it back. When false (default), `target` is clamped on every
    /// write.
    pub overscroll: bool,
    /// Scrollbar fade alpha in `[0, 1]`. Pinned to 1 while the spring is
    /// chasing, while pointer is over the bar, or while a thumb is being
    /// dragged; decays over `style.fade_seconds` once idle. flatten emits
    /// the bars at `inst.color.a *= bar_alpha` so they fade in/out
    /// without a separate timeline.
    pub bar_alpha: f32,
    /// Visual + behavior config for both bars.
    pub style: ScrollbarStyle,
    /// Per-axis pointer hover state: `[x, y]`. Set by the input layer
    /// when the cursor enters the bar's track AABB; read by emit to
    /// pick the thumb color and pin `bar_alpha`. `[X, Y]` indexed by
    /// `ScrollAxis::index`.
    pub bar_hover: [bool; 2],
    /// Per-axis active (mouse-down on thumb) state. While true the
    /// thumb paints at `style.thumb_active_color` and the bar can't
    /// fade out.
    pub bar_active: [bool; 2],
}

impl Default for ScrollState {
    fn default() -> Self {
        Self {
            current: [0.0; 2],
            target: [0.0; 2],
            stiffness: 12.0,
            overscroll: false,
            bar_alpha: 0.0,
            style: ScrollbarStyle::default(),
            bar_hover: [false; 2],
            bar_active: [false; 2],
        }
    }
}

impl ScrollState {
    /// True while at least one axis is being dragged. Used to gate
    /// "pointer-down on track to jump" — clicks during a drag should
    /// not retarget.
    pub fn dragging(&self) -> bool {
        self.bar_active[0] || self.bar_active[1]
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub style: ShapeStyle,
    pub layout: LayoutStyle,
    /// Post-layout absolute rect `[x, y, w, h]`. Written by
    /// [`crate::layout::compute_layout`]; read by `flatten_with_text`.
    pub rect: [f32; 4],
    /// Bounding extent of all children, in physical px relative to
    /// `rect.xy`. Populated by `compute_layout` for every container;
    /// used by scroll math (`max_offset = content_size - rect_size`).
    pub content_size: [f32; 2],
    /// Present iff `layout.scrolls()`. See [`ScrollState`].
    pub scroll: Option<ScrollState>,
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
                content_size: [0.0; 2],
                scroll: None,
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

    pub fn overflow(mut self, ox: crate::layout::Overflow, oy: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_x = ox;
        self.node.layout.overflow_y = oy;
        self
    }

    pub fn overflow_x(mut self, o: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_x = o;
        self
    }

    pub fn overflow_y(mut self, o: crate::layout::Overflow) -> Self {
        self.node.layout.overflow_y = o;
        self
    }

    pub fn scroll(self) -> Self {
        self.overflow(crate::layout::Overflow::Scroll, crate::layout::Overflow::Scroll)
    }

    pub fn scroll_x(self) -> Self {
        self.overflow_x(crate::layout::Overflow::Scroll)
    }

    pub fn scroll_y(self) -> Self {
        self.overflow_y(crate::layout::Overflow::Scroll)
    }

    pub fn clip(self) -> Self {
        self.overflow(crate::layout::Overflow::Hidden, crate::layout::Overflow::Hidden)
    }

    /// Spring stiffness for scroll smoothing. Stored on the node's
    /// pre-allocated `ScrollState`; only takes effect once the node is
    /// also marked scrollable on at least one axis (otherwise insert
    /// drops `scroll` to `None`).
    pub fn scroll_smoothness(mut self, k: f32) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.stiffness = k.max(0.0);
        self
    }

    pub fn overscroll(mut self, on: bool) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.overscroll = on;
        self
    }

    /// Replace the entire scrollbar style on this node. Allocates a
    /// `ScrollState` so the style sticks even if the node isn't yet
    /// scrollable; insert reconciles the `scrollable_ids` index.
    pub fn scrollbar_style(mut self, style: ScrollbarStyle) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.style = style;
        self
    }

    /// Mutate the scrollbar style with a closure: e.g.
    /// `.scrollbar(|s| s.thickness(8.0).thumb_color([1,1,1,0.7]))`.
    pub fn scrollbar<F: FnOnce(ScrollbarStyle) -> ScrollbarStyle>(mut self, f: F) -> Self {
        let s = self.node.scroll.get_or_insert_with(ScrollState::default);
        s.style = f(s.style);
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
    /// Per-glass frosted-texture variation (logical px). Per-fragment
    /// hash scatters the backdrop sample by this many pixels at the
    /// chosen mip. 0 = mirror-smooth; ~1 = subtle frost; ~3 = pebbled.
    pub fn roughness(mut self, px: f32) -> Self {
        self.node.style.roughness = px;
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
    /// Every node that currently owns a `ScrollState` (overflow set to
    /// Scroll on at least one axis). Used by `tick_scrolls` so the
    /// frame loop doesn't have to re-walk the tree every tick.
    /// Maintained by `set_layout_overflow` / `remove`.
    scrollable_ids: Vec<NodeId>,
}

impl NodeTree {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, mut node: Node) -> NodeId {
        let is_glass = matches!(node.style.kind, ShapeKind::Glass);
        // Reconcile scroll state with layout overflow declared on the
        // builder side: if either axis is Scroll, ensure ScrollState
        // exists so `scrollable_ids` and the wheel/tick paths see it.
        let needs_scroll = node.layout.scrolls();
        if needs_scroll && node.scroll.is_none() {
            node.scroll = Some(ScrollState::default());
        } else if !needs_scroll && node.scroll.is_some() {
            node.scroll = None;
        }
        let has_scroll = node.scroll.is_some();
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
        if has_scroll {
            self.scrollable_ids.push(id);
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
        let payload = slot.payload.as_ref();
        let was_glass = payload
            .map(|n| matches!(n.style.kind, ShapeKind::Glass))
            .unwrap_or(false);
        let was_scrollable = payload.map(|n| n.scroll.is_some()).unwrap_or(false);
        slot.generation = slot.generation.wrapping_add(1);
        slot.payload = None;
        self.free.push(id.index);
        self.roots.retain(|r| *r != id);
        if was_scrollable {
            self.scrollable_ids.retain(|sid| *sid != id);
        }
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

    /// Set per-axis overflow. Allocates `ScrollState` on the node when
    /// either axis becomes Scroll; clears it when both axes drop back
    /// to Visible/Hidden. Maintains `scrollable_ids` so the frame
    /// loop's scroll tick has an O(1) iteration list.
    pub fn set_layout_overflow(&mut self, id: NodeId, ox: crate::layout::Overflow,
                               oy: crate::layout::Overflow) {
        use crate::layout::Overflow;
        let mask = self.transform_mask();
        let mut allocated = false;
        let mut cleared = false;
        if let Some(n) = self.get_mut_raw(id) {
            let changed = n.layout.overflow_x != ox || n.layout.overflow_y != oy;
            if !changed {
                return;
            }
            n.layout.overflow_x = ox;
            n.layout.overflow_y = oy;
            let needs_scroll = matches!(ox, Overflow::Scroll) || matches!(oy, Overflow::Scroll);
            match (needs_scroll, &n.scroll) {
                (true, None) => {
                    n.scroll = Some(ScrollState::default());
                    allocated = true;
                }
                (false, Some(_)) => {
                    n.scroll = None;
                    cleared = true;
                }
                _ => {}
            }
            self.dirty |= mask;
        }
        if allocated {
            self.scrollable_ids.push(id);
        }
        if cleared {
            self.scrollable_ids.retain(|sid| *sid != id);
        }
    }

    /// Advance every active scroll spring by `dt` seconds. Spring is a
    /// single-pole exponential ease toward `target`: `current += (target
    /// - current) * (1 - exp(-stiffness * dt))`. Snaps when within
    /// 0.5 px so the loop can park on `Wait`. Returns true when at
    /// least one node moved — caller flags the dirty mask + flushes.
    /// Sets `TRANSFORM` (and `BACKDROP` if glass exists) so flatten
    /// + the blur pass re-run with the new offsets.
    pub fn tick_scrolls(&mut self, dt: f32) -> bool {
        if self.scrollable_ids.is_empty() || dt <= 0.0 {
            return false;
        }
        let mut moved = false;
        let mut bar_changed = false;
        for i in 0..self.scrollable_ids.len() {
            let id = self.scrollable_ids[i];
            let Some(slot) = self.slots.get_mut(id.index as usize) else {
                continue;
            };
            if slot.generation != id.generation {
                continue;
            }
            let Some(n) = slot.payload.as_mut() else {
                continue;
            };
            let Some(s) = n.scroll.as_mut() else {
                continue;
            };
            // Spring step.
            if s.current != s.target {
                let alpha = 1.0 - (-s.stiffness * dt).exp();
                let mut new = [
                    s.current[0] + (s.target[0] - s.current[0]) * alpha,
                    s.current[1] + (s.target[1] - s.current[1]) * alpha,
                ];
                if (s.target[0] - new[0]).abs() < 0.5 {
                    new[0] = s.target[0];
                }
                if (s.target[1] - new[1]).abs() < 0.5 {
                    new[1] = s.target[1];
                }
                if new != s.current {
                    s.current = new;
                    moved = true;
                }
                // Hold the bar fully visible while chasing.
                s.bar_alpha = 1.0;
            }
            // Hold visible whenever the user is interacting with the
            // bar or the style demands always-on. Otherwise drain.
            let hold = s.style.always_visible
                || s.bar_hover[0]
                || s.bar_hover[1]
                || s.bar_active[0]
                || s.bar_active[1]
                || s.current != s.target;
            if hold {
                if s.bar_alpha < 1.0 {
                    s.bar_alpha = 1.0;
                    bar_changed = true;
                }
            } else if s.bar_alpha > 0.0 {
                let step = if s.style.fade_seconds > 0.0 {
                    dt / s.style.fade_seconds
                } else {
                    1.0
                };
                let new_alpha = (s.bar_alpha - step).max(0.0);
                if new_alpha != s.bar_alpha {
                    s.bar_alpha = new_alpha;
                    bar_changed = true;
                }
            }
        }
        if moved || bar_changed {
            self.dirty |= self.scroll_mask();
        }
        moved || bar_changed
    }

    /// True when at least one scrollable node still needs another tick:
    /// either the spring is chasing (`current != target`) or the bar is
    /// mid-fade (`bar_alpha > 0` while idle). Drives the loop's
    /// `WaitUntil` scheduling so the bar fades cleanly to 0 before the
    /// loop parks on `Wait`.
    pub fn has_active_scrolls(&self) -> bool {
        self.scrollable_ids.iter().any(|&id| {
            self.get(id)
                .and_then(|n| n.scroll.as_ref())
                .map(|s| {
                    s.current != s.target
                        || s.bar_alpha > 0.0
                        || s.style.always_visible
                        || s.bar_hover[0]
                        || s.bar_hover[1]
                        || s.bar_active[0]
                        || s.bar_active[1]
                })
                .unwrap_or(false)
        })
    }

    /// Set the scroll target (where the spring is easing toward) on a
    /// scrollable node. Clamped to `[0, content_size - rect_size]`
    /// unless `ScrollState.overscroll == true`. Per-axis overflow
    /// gates the write — non-scroll axes ignore the input. Bumps
    /// TRANSFORM when the target moves so the next flush ticks the
    /// spring.
    pub fn set_scroll_target(&mut self, id: NodeId, target: [f32; 2]) {
        let (rect, content, sx, sy) = match self.get(id) {
            Some(n) => (
                n.rect,
                n.content_size,
                n.layout.overflow_x.scrolls(),
                n.layout.overflow_y.scrolls(),
            ),
            None => return,
        };
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let max_off_x = (content[0] - rect[2]).max(0.0);
            let max_off_y = (content[1] - rect[3]).max(0.0);
            let want_x = if sx { target[0] } else { s.target[0] };
            let want_y = if sy { target[1] } else { s.target[1] };
            let new_target = if s.overscroll {
                [want_x, want_y]
            } else {
                [want_x.clamp(0.0, max_off_x), want_y.clamp(0.0, max_off_y)]
            };
            if s.target != new_target {
                s.target = new_target;
                if !s.style.auto_hide {
                    s.bar_alpha = 1.0;
                }
                self.dirty |= mask;
            }
        }
    }

    /// Add to the scroll target. Convenience for wheel input — caller
    /// passes raw delta and clamping happens here. Per-axis overflow
    /// gates the write: a Scroll-x-only container ignores y delta even
    /// if its `content_size.y > rect.h`. Returns the actual delta
    /// applied (may be less than requested at edges or zero on a non-
    /// scroll axis) so a wheel dispatcher can bubble the remainder to
    /// a parent scroll ancestor.
    pub fn add_scroll_delta(&mut self, id: NodeId, delta: [f32; 2]) -> [f32; 2] {
        let (rect, content, sx, sy) = match self.get(id) {
            Some(n) => (
                n.rect,
                n.content_size,
                n.layout.overflow_x.scrolls(),
                n.layout.overflow_y.scrolls(),
            ),
            None => return [0.0; 2],
        };
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let max_off_x = (content[0] - rect[2]).max(0.0);
            let max_off_y = (content[1] - rect[3]).max(0.0);
            let want_x = s.target[0] + if sx { delta[0] } else { 0.0 };
            let want_y = s.target[1] + if sy { delta[1] } else { 0.0 };
            let new_target = if s.overscroll {
                [want_x, want_y]
            } else {
                [want_x.clamp(0.0, max_off_x), want_y.clamp(0.0, max_off_y)]
            };
            let applied = [new_target[0] - s.target[0], new_target[1] - s.target[1]];
            if applied != [0.0, 0.0] {
                s.target = new_target;
                if !s.style.auto_hide {
                    s.bar_alpha = 1.0;
                }
                self.dirty |= mask;
            }
            return applied;
        }
        [0.0; 2]
    }

    /// Read the displayed scroll offset (current, not target). Returns
    /// `[0, 0]` for non-scrollable nodes.
    pub fn scroll_offset(&self, id: NodeId) -> [f32; 2] {
        self.get(id)
            .and_then(|n| n.scroll.as_ref())
            .map(|s| s.current)
            .unwrap_or([0.0; 2])
    }

    /// Read content size (bounding extent of children, includes
    /// trailing padding). Returns the node's own `rect` size for
    /// non-container leaves.
    pub fn scrollable_size(&self, id: NodeId) -> [f32; 2] {
        self.get(id).map(|n| n.content_size).unwrap_or([0.0; 2])
    }

    /// Set the spring stiffness (ease rate). No-op on non-scrollable.
    pub fn set_scroll_stiffness(&mut self, id: NodeId, k: f32) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.stiffness = k.max(0.0);
        }
    }

    pub fn set_scroll_overscroll(&mut self, id: NodeId, on: bool) {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            s.overscroll = on;
        }
    }

    /// Replace the entire scrollbar style on `id`. Allocates a
    /// `ScrollState` if one isn't already present so style changes can
    /// be authored before `.scroll()` is called.
    pub fn set_scrollbar_style(&mut self, id: NodeId, style: ScrollbarStyle) {
        let mut allocated = false;
        if let Some(n) = self.get_mut_raw(id) {
            if n.scroll.is_none() {
                n.scroll = Some(ScrollState::default());
                allocated = true;
            }
            if let Some(s) = n.scroll.as_mut() {
                s.style = style;
            }
        }
        if allocated {
            // Only push to scrollable_ids if the node already declared
            // an overflow that scrolls — otherwise insert/remove
            // already manages it. We allocate eagerly so styles can be
            // set before .scroll(); insert reconciles on add.
            let scrolls = self.get(id).map(|n| n.layout.scrolls()).unwrap_or(false);
            if scrolls && !self.scrollable_ids.contains(&id) {
                self.scrollable_ids.push(id);
            }
        }
    }

    /// Mutate the existing scrollbar style in place. Same allocation
    /// rules as [`Self::set_scrollbar_style`].
    pub fn with_scrollbar_style<F: FnOnce(&mut ScrollbarStyle)>(&mut self, id: NodeId, f: F) {
        let mut allocated = false;
        if let Some(n) = self.get_mut_raw(id) {
            if n.scroll.is_none() {
                n.scroll = Some(ScrollState::default());
                allocated = true;
            }
            if let Some(s) = n.scroll.as_mut() {
                f(&mut s.style);
            }
        }
        if allocated {
            let scrolls = self.get(id).map(|n| n.layout.scrolls()).unwrap_or(false);
            if scrolls && !self.scrollable_ids.contains(&id) {
                self.scrollable_ids.push(id);
            }
        }
    }

    /// Set per-axis pointer-hover flags on a scrollable node. Returns
    /// true if anything changed (caller can use this to gate redraw).
    /// `[X, Y]` indexed by `ScrollAxis::index`.
    pub fn set_bar_hover(&mut self, id: NodeId, hover: [bool; 2]) -> bool {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
            && s.bar_hover != hover
        {
            s.bar_hover = hover;
            // Hovering pops the bar to full alpha immediately so the
            // user gets feedback without waiting on the next tick.
            if hover[0] || hover[1] {
                s.bar_alpha = 1.0;
            }
            // SCROLL only — bar color change re-flattens but doesn't
            // touch layout or the opaque backdrop.
            self.dirty |= dirty::SCROLL;
            return true;
        }
        false
    }

    /// Set per-axis active (mouse-down on thumb) flags. Returns true
    /// on change.
    pub fn set_bar_active(&mut self, id: NodeId, active: [bool; 2]) -> bool {
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
            && s.bar_active != active
        {
            s.bar_active = active;
            if active[0] || active[1] {
                s.bar_alpha = 1.0;
            }
            self.dirty |= dirty::SCROLL;
            return true;
        }
        false
    }

    /// Snap scroll on one axis to `pos` immediately (no spring chase).
    /// Intended for thumb-drag — the pointer is the authoritative
    /// position so easing toward it would just lag behind. Writes
    /// both `current` and `target` so the spring stays at rest.
    /// Clamped via the same overscroll rules as `set_scroll_target`.
    pub fn set_scroll_immediate(&mut self, id: NodeId, axis: ScrollAxis, pos: f32) {
        let (rect, content) = match self.get(id) {
            Some(n) => (n.rect, n.content_size),
            None => return,
        };
        let mask = self.scroll_mask();
        if let Some(n) = self.get_mut_raw(id)
            && let Some(s) = n.scroll.as_mut()
        {
            let i = axis.index();
            let max_off = (content[i] - rect[2 + i]).max(0.0);
            let new_pos = if s.overscroll { pos } else { pos.clamp(0.0, max_off) };
            if (s.current[i] - new_pos).abs() > f32::EPSILON
                || (s.target[i] - new_pos).abs() > f32::EPSILON
            {
                s.current[i] = new_pos;
                s.target[i] = new_pos;
                s.bar_alpha = 1.0;
                self.dirty |= mask;
            }
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

    /// Mask for scroll-offset writes. Layout doesn't need to re-run
    /// (rects are unchanged), only flatten — so `SCROLL` instead of
    /// `TRANSFORM`. Backdrop still re-blurs when glass exists because
    /// opaque content under glass moved.
    fn scroll_mask(&self) -> u32 {
        if self.has_glass() {
            dirty::SCROLL | dirty::BACKDROP
        } else {
            dirty::SCROLL
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
    ///
    /// Clip + scroll offset propagate down the tree. Each node receives
    /// the intersection of its ancestors' clipping rects and the sum of
    /// its ancestors' scroll offsets — the recursive walk maintains the
    /// stack implicitly so emitted instances/hits are already in screen
    /// space.
    pub fn flatten(
        &self,
        scale: f32,
    ) -> (Vec<FlatEvent>, Vec<HitEntry>, Vec<ScrollHit>, Vec<ScrollbarHit>) {
        let mut events = Vec::with_capacity(self.len());
        let mut hits = Vec::new();
        let mut scroll_hits = Vec::new();
        let mut scroll_bars = Vec::new();
        self.flatten_into_buffers(
            scale,
            &mut events,
            &mut hits,
            &mut scroll_hits,
            &mut scroll_bars,
        );
        (events, hits, scroll_hits, scroll_bars)
    }

    /// Same as [`Self::flatten`] but reuses caller-owned buffers
    /// instead of allocating fresh `Vec`s. Each buffer is `clear()`ed
    /// before population so callers can amortize allocation across
    /// frames (a steady-state scene reuses the same heap blocks every
    /// flatten — saves ~5–20µs of allocator churn per frame). Hits are
    /// reversed at the end so the cache reads topmost-first as usual.
    pub fn flatten_into_buffers(
        &self,
        scale: f32,
        events: &mut Vec<FlatEvent>,
        hits: &mut Vec<HitEntry>,
        scroll_hits: &mut Vec<ScrollHit>,
        scroll_bars: &mut Vec<ScrollbarHit>,
    ) {
        events.clear();
        hits.clear();
        scroll_hits.clear();
        scroll_bars.clear();
        let mut scroll_stack: Vec<NodeId> = Vec::new();
        for root in &self.roots {
            self.flatten_into(
                *root,
                1.0,
                NO_CLIP,
                [0.0; 2],
                &mut scroll_stack,
                events,
                hits,
                scroll_hits,
                scroll_bars,
                scale,
            );
        }
        hits.reverse();
    }

    #[cfg(test)]
    fn dirty_for_test(&self) -> u32 {
        self.dirty
    }

    #[allow(clippy::too_many_arguments)]
    fn flatten_into(
        &self,
        id: NodeId,
        parent_opacity: f32,
        clip: [f32; 4],
        offset: [f32; 2],
        scroll_stack: &mut Vec<NodeId>,
        events: &mut Vec<FlatEvent>,
        hits: &mut Vec<HitEntry>,
        scroll_hits: &mut Vec<ScrollHit>,
        scroll_bars: &mut Vec<ScrollbarHit>,
        scale: f32,
    ) {
        let Some(node) = self.get(id) else { return };
        if !node.visible {
            return;
        }
        let rect = node.rect;
        let abs = [rect[0] - offset[0], rect[1] - offset[1]];
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
                    clip_rect: clip,
                    position: abs,
                    size,
                    shadow_offset: node.style.shadow_offset,
                    shape_kind: node.style.kind.as_u32(),
                    roughness: node.style.roughness,
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
                        clip_rect: clip,
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
                        clip_rect: clip,
                    }));
                }
            }
        }
        if node.interact.is_any() || node.window_action.is_some() {
            hits.push(HitEntry {
                node_id: id,
                bounds: [abs[0], abs[1], abs[0] + size[0], abs[1] + size[1]],
                clip_rect: clip,
            });
        }
        // Emit a ScrollHit for any container whose layout scrolls. The
        // ancestor chain is innermost-first: this node first, then each
        // scroll ancestor outward. Wheel routing pops from the front
        // when bubbling at edges.
        let pushed_scroll = if node.scroll.is_some() && node.layout.scrolls() {
            let mut chain = Vec::with_capacity(scroll_stack.len() + 1);
            chain.push(id);
            chain.extend(scroll_stack.iter().rev().copied());
            scroll_hits.push(ScrollHit {
                node_id: id,
                bounds: [abs[0], abs[1], abs[0] + size[0], abs[1] + size[1]],
                clip_rect: clip,
                ancestor_chain: chain,
            });
            scroll_stack.push(id);
            true
        } else {
            false
        };
        // Children: intersect parent clip with this node's self-clip
        // (axis-aware — only narrow the axes that clip), then add this
        // node's scroll offset to the running offset.
        let child_clip = if node.layout.clips() {
            let self_clip = [
                if node.layout.overflow_x.clips() { abs[0] } else { -1.0e30 },
                if node.layout.overflow_y.clips() { abs[1] } else { -1.0e30 },
                if node.layout.overflow_x.clips() { abs[0] + size[0] } else { 1.0e30 },
                if node.layout.overflow_y.clips() { abs[1] + size[1] } else { 1.0e30 },
            ];
            intersect_clip(clip, self_clip)
        } else {
            clip
        };
        let child_offset = if let Some(s) = node.scroll.as_ref() {
            [offset[0] + s.current[0], offset[1] + s.current[1]]
        } else {
            offset
        };
        for &child in &node.children {
            self.flatten_into(
                child,
                opacity,
                child_clip,
                child_offset,
                scroll_stack,
                events,
                hits,
                scroll_hits,
                scroll_bars,
                scale,
            );
        }
        // Emit scrollbar geometry last so visible bars paint over
        // children. The bar lives at the container's *unscrolled*
        // position (uses `abs`, not `child_offset`) and inherits the
        // parent's clip. Hits are populated regardless of `bar_alpha`
        // so input can detect hover-enter on a faded-out bar's region.
        if let Some(s) = node.scroll.as_ref() {
            emit_scrollbars(
                id,
                node,
                s,
                abs,
                size,
                opacity,
                clip,
                scale,
                events,
                scroll_bars,
            );
        }
        if pushed_scroll {
            scroll_stack.pop();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_scrollbars(
    node_id: NodeId,
    node: &Node,
    s: &ScrollState,
    abs: [f32; 2],
    size: [f32; 2],
    opacity: f32,
    clip: [f32; 4],
    scale: f32,
    events: &mut Vec<FlatEvent>,
    scroll_bars: &mut Vec<ScrollbarHit>,
) {
    let style = &s.style;
    let bar_w = style.thickness * scale;
    let bar_margin = style.margin * scale;
    let min_thumb = style.min_thumb * scale;
    let bar_alpha = if style.always_visible { 1.0 } else { s.bar_alpha };
    let visual = bar_alpha * opacity;

    let mut emit_quad = |position: [f32; 2], box_size: [f32; 2], rgba: [f32; 4]| {
        if rgba[3] <= 0.001 || box_size[0] <= 0.0 || box_size[1] <= 0.0 {
            return;
        }
        events.push(FlatEvent::Shape(ShapeInstance {
            color: rgba,
            border_color: [0.0; 4],
            shadow_color: [0.0; 4],
            // Logical px — `expand_events_into` re-scales it.
            border_radius: [style.radius; 4],
            backdrop_uv_rect: [0.0; 4],
            clip_rect: clip,
            position,
            size: box_size,
            shadow_offset: [0.0; 2],
            shape_kind: SHAPE_KIND_RECT,
            roughness: 0.0,
            border_width: 0.0,
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
        }));
    };

    // Y bar.
    if node.layout.overflow_y.scrolls() {
        let max_off = (node.content_size[1] - size[1]).max(0.0);
        if max_off > 0.0 {
            let track_x = match style.y_side {
                BarSide::End => abs[0] + size[0] - bar_w - bar_margin,
                BarSide::Start => abs[0] + bar_margin,
            };
            let track_y = abs[1] + bar_margin;
            let track_h = size[1] - bar_margin * 2.0;
            if track_h > 0.0 {
                let visible_ratio = (size[1] / node.content_size[1]).clamp(0.0, 1.0);
                let thumb_h = (track_h * visible_ratio).max(min_thumb).min(track_h);
                let frac = (s.current[1] / max_off).clamp(0.0, 1.0);
                let thumb_y = track_y + frac * (track_h - thumb_h);
                let thumb_color = pick_thumb_color(style, s.bar_active[1], s.bar_hover[1]);
                let track_rgba = scale_alpha(style.track_color, visual);
                let thumb_rgba = scale_alpha(thumb_color, visual);
                emit_quad([track_x, track_y], [bar_w, track_h], track_rgba);
                emit_quad([track_x, thumb_y], [bar_w, thumb_h], thumb_rgba);
                scroll_bars.push(ScrollbarHit {
                    node_id,
                    axis: ScrollAxis::Y,
                    track: [track_x, track_y, track_x + bar_w, track_y + track_h],
                    thumb: [track_x, thumb_y, track_x + bar_w, thumb_y + thumb_h],
                    clip_rect: clip,
                    max_offset: max_off,
                    track_travel: (track_h - thumb_h).max(0.0),
                });
            }
        }
    }
    // X bar.
    if node.layout.overflow_x.scrolls() {
        let max_off = (node.content_size[0] - size[0]).max(0.0);
        if max_off > 0.0 {
            let track_x = abs[0] + bar_margin;
            let track_y = match style.x_side {
                BarSide::End => abs[1] + size[1] - bar_w - bar_margin,
                BarSide::Start => abs[1] + bar_margin,
            };
            let track_w = size[0] - bar_margin * 2.0;
            // If both axes scroll, leave space for the y-bar on its
            // chosen side so the two tracks don't visually overlap.
            let reserved = if node.layout.overflow_y.scrolls() {
                bar_w + bar_margin
            } else {
                0.0
            };
            let (track_x, track_w) = match (node.layout.overflow_y.scrolls(), style.y_side) {
                (true, BarSide::Start) => (track_x + reserved, track_w - reserved),
                (true, BarSide::End) => (track_x, track_w - reserved),
                _ => (track_x, track_w),
            };
            if track_w > 0.0 {
                let visible_ratio = (size[0] / node.content_size[0]).clamp(0.0, 1.0);
                let thumb_w = (track_w * visible_ratio).max(min_thumb).min(track_w);
                let frac = (s.current[0] / max_off).clamp(0.0, 1.0);
                let thumb_x = track_x + frac * (track_w - thumb_w);
                let thumb_color = pick_thumb_color(style, s.bar_active[0], s.bar_hover[0]);
                let track_rgba = scale_alpha(style.track_color, visual);
                let thumb_rgba = scale_alpha(thumb_color, visual);
                emit_quad([track_x, track_y], [track_w, bar_w], track_rgba);
                emit_quad([thumb_x, track_y], [thumb_w, bar_w], thumb_rgba);
                scroll_bars.push(ScrollbarHit {
                    node_id,
                    axis: ScrollAxis::X,
                    track: [track_x, track_y, track_x + track_w, track_y + bar_w],
                    thumb: [thumb_x, track_y, thumb_x + thumb_w, track_y + bar_w],
                    clip_rect: clip,
                    max_offset: max_off,
                    track_travel: (track_w - thumb_w).max(0.0),
                });
            }
        }
    }
}

fn pick_thumb_color(style: &ScrollbarStyle, active: bool, hover: bool) -> [f32; 4] {
    if active {
        style.thumb_active_color
    } else if hover {
        style.thumb_hover_color
    } else {
        style.thumb_color
    }
}

fn scale_alpha(c: [f32; 4], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], c[3] * a]
}

fn intersect_clip(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
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
    fn scroll_state_allocates_on_overflow_scroll() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().build());
        assert!(t.get(id).unwrap().scroll.is_none());
        t.set_layout_overflow(id, crate::layout::Overflow::Scroll, crate::layout::Overflow::Visible);
        assert!(t.get(id).unwrap().scroll.is_some());
    }

    #[test]
    fn add_scroll_delta_clamps_and_reports_remainder() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        // content > rect → 100 px scroll budget on y.
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 100.0];
            n.content_size = [200.0, 200.0];
        }
        let applied = t.add_scroll_delta(id, [0.0, 200.0]);
        assert!((applied[1] - 100.0).abs() < 0.01, "clamped applied = {applied:?}");
        // Already at edge — next push should report zero applied so
        // wheel routing can bubble.
        let again = t.add_scroll_delta(id, [0.0, 50.0]);
        assert_eq!(again, [0.0, 0.0]);
    }

    #[test]
    fn add_scroll_delta_ignores_non_scroll_axis() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_x().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [400.0, 400.0];
        }
        // y has plenty of content but isn't a scroll axis — should be 0.
        let applied = t.add_scroll_delta(id, [0.0, 50.0]);
        assert_eq!(applied, [0.0, 0.0]);
    }

    #[test]
    fn tick_scrolls_eases_toward_target_and_snaps() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 100.0];
            n.content_size = [200.0, 1100.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 1000.0]);
        // ~2 sec of 60 Hz ticks at default stiffness 12 → spring snaps
        // within the first half-second, then bar_alpha (default 0.8 s
        // fade) drains to 0 — has_active_scrolls returns false only
        // after both are settled.
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            t.tick_scrolls(dt);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current, s.target, "should have snapped");
        assert_eq!(s.bar_alpha, 0.0, "bar fade should have drained");
        assert!(!t.has_active_scrolls());
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

    #[test]
    fn flatten_emits_scrollbar_hits() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 200.0, 200.0];
            n.content_size = [800.0, 1000.0];
            // Force-show the bars regardless of fade — we want geometry.
            if let Some(s) = n.scroll.as_mut() {
                s.bar_alpha = 1.0;
            }
        }
        let (_events, _hits, _scroll_hits, bars) = t.flatten(1.0);
        assert_eq!(bars.len(), 2, "two bars (X + Y) expected");
        let x = bars.iter().find(|b| b.axis == ScrollAxis::X).unwrap();
        let y = bars.iter().find(|b| b.axis == ScrollAxis::Y).unwrap();
        assert!(x.track_travel > 0.0);
        assert!(y.track_travel > 0.0);
        assert_eq!(x.max_offset, 800.0 - 200.0);
        assert_eq!(y.max_offset, 1000.0 - 200.0);
    }

    #[test]
    fn always_visible_keeps_alpha_pinned() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .scrollbar(|s| s.always_visible(true))
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        // No movement at all — but the tick should still pin alpha.
        for _ in 0..30 {
            t.tick_scrolls(1.0 / 60.0);
        }
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_alpha, 1.0, "always_visible must hold alpha at 1");
        assert!(t.has_active_scrolls(), "always_visible keeps loop ticking");
    }

    #[test]
    fn auto_hide_skips_pop_on_movement() {
        let mut t = NodeTree::new();
        let id = t.add_root(
            Node::rect()
                .scroll_y()
                .scrollbar(|s| s.auto_hide(true))
                .build(),
        );
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 1000.0];
        }
        let _ = t.add_scroll_delta(id, [0.0, 100.0]);
        let s = t.get(id).unwrap().scroll.unwrap();
        // auto_hide: target moved but bar should still be invisible.
        assert_eq!(s.bar_alpha, 0.0, "auto_hide must not pop on scroll");
    }

    #[test]
    fn bar_hover_pops_alpha_to_one() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        assert_eq!(t.get(id).unwrap().scroll.unwrap().bar_alpha, 0.0);
        let changed = t.set_bar_hover(id, [false, true]);
        assert!(changed);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.bar_alpha, 1.0);
        assert_eq!(s.bar_hover, [false, true]);
    }

    #[test]
    fn set_scroll_immediate_writes_both_current_and_target() {
        let mut t = NodeTree::new();
        let id = t.add_root(Node::rect().scroll_y().build());
        if let Some(n) = t.get_mut_raw(id) {
            n.rect = [0.0, 0.0, 100.0, 100.0];
            n.content_size = [100.0, 500.0];
        }
        t.set_scroll_immediate(id, ScrollAxis::Y, 200.0);
        let s = t.get(id).unwrap().scroll.unwrap();
        assert_eq!(s.current[1], 200.0);
        assert_eq!(s.target[1], 200.0, "drag must keep spring at rest");
    }
}
