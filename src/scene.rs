//! Nested scene builder.
//!
//! `Scene` is a thin wrapper around [`NodeTree`] that hides parent-id
//! threading: nested scopes are introduced via `child(|p| { … })`
//! closures, and node handles are returned implicitly through the
//! builder chain (or looked up by name via [`SceneCtx::node`]).
//!
//! The builder also accepts reactive props through [`Bind<T>`]. When
//! a bind is reactive (signal/computed/animated), it gets recorded
//! into a [`BindRegistry`] so the [`crate::app::App`] shell can
//! re-evaluate it on every signal change and drive auto-tweens.
//! Plain `Value` binds are written into the node once and dropped.
//!
//! The builder API mirrors `Node::rect()` so existing demos can be
//! ported by mostly structural rewrites — the only real difference
//! is that mutation goes through the live tree rather than a
//! detached `Node` value.

use std::collections::HashMap;

use crate::node::{Node, NodeId, NodeInteract, NodeTree, ShapeKind};
use crate::reactive::Bind;
use crate::signal::Signal;

/// Shared mutable state passed through every scene builder call.
/// Owned by the [`crate::app::App`] shell; user code only sees a
/// `&mut Scene` borrowing into it.
pub struct SceneCtx {
    pub tree: NodeTree,
    pub names: HashMap<String, NodeId>,
    pub binds: BindRegistry,
}

impl Default for SceneCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCtx {
    pub fn new() -> Self {
        Self {
            tree: NodeTree::new(),
            names: HashMap::new(),
            binds: BindRegistry::default(),
        }
    }

    /// Look up a previously-named node. Useful for late wiring (e.g.
    /// pulling a handle from outside the scene closure) and for
    /// debug overlays.
    pub fn node(&self, name: &str) -> Option<NodeId> {
        self.names.get(name).copied()
    }
}

/// Per-prop reactive bind storage. The shell walks each list on
/// every event/anim tick and, on a version bump, either snaps the
/// node prop or starts an auto-tween toward the new target.
///
/// Stage 1 only tracks `color`. Position/size/opacity binds will
/// land alongside the layout engine in stage 2.
#[derive(Default)]
pub struct BindRegistry {
    pub color: Vec<ColorBindSlot>,
}

pub struct ColorBindSlot {
    pub node_id: NodeId,
    pub bind: Bind<[f32; 4]>,
    pub last_version: u64,
    /// For animated binds: the per-slot signal that the timeline
    /// drives. Each tick the shell pushes its current value through
    /// `tree.set_color`. `None` for non-animated binds (snap mode).
    pub displayed: Option<Signal<[f32; 4]>>,
}

/// A scoped scene cursor. Holds an implicit `parent` so nested
/// `child` closures don't need to thread `NodeId` by hand.
pub struct Scene<'a> {
    ctx: &'a mut SceneCtx,
    parent: Option<NodeId>,
}

impl<'a> Scene<'a> {
    /// Construct a top-level scene. The root closure receives this
    /// as `&mut Scene`; nested children come from `child(|p| …)`.
    pub fn root(ctx: &'a mut SceneCtx) -> Self {
        Self { ctx, parent: None }
    }

    pub fn ctx(&self) -> &SceneCtx {
        self.ctx
    }

    pub fn ctx_mut(&mut self) -> &mut SceneCtx {
        self.ctx
    }

    /// Add a rect child under the current parent. `name` is optional
    /// and indexes into `SceneCtx::node`; pass `""` for an anonymous
    /// node.
    pub fn rect(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), ShapeKind::Rect)
    }

    /// Add a frosted glass child under the current parent.
    pub fn glass(&mut self, name: impl Into<String>) -> NodeBuilderRef<'_> {
        self.spawn(name.into(), ShapeKind::Glass)
    }

    fn spawn(&mut self, name: String, kind: ShapeKind) -> NodeBuilderRef<'_> {
        let mut node = match kind {
            ShapeKind::Rect => Node::rect().build(),
            ShapeKind::Glass => Node::glass().build(),
        };
        // `Node::rect/glass` set defaults; nothing else to override here.
        let _ = &mut node;
        let id = match self.parent {
            Some(p) => self.ctx.tree.add_child(p, node),
            None => self.ctx.tree.add_root(node),
        };
        if !name.is_empty() {
            self.ctx.names.insert(name, id);
        }
        NodeBuilderRef {
            ctx: self.ctx,
            id,
        }
    }
}

/// Live-tree builder ref. Each method mutates the inserted node
/// directly through `SceneCtx::tree`. Reactive props register a
/// slot in `BindRegistry`. Returns `&mut Self` so chains are simple
/// and don't need a terminal `.build()`.
pub struct NodeBuilderRef<'a> {
    ctx: &'a mut SceneCtx,
    id: NodeId,
}

impl<'a> NodeBuilderRef<'a> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn pos(&mut self, x: f32, y: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.position = [x, y];
        }
        self
    }

    pub fn size(&mut self, w: f32, h: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.size = [w, h];
        }
        self
    }

    /// Set the fill color. Accepts a raw `[f32; 4]`, a `Signal`, a
    /// `Computed`, or an `animated(...)` wrapper — anything that
    /// converts into `Bind<[f32; 4]>`. Reactive variants are
    /// registered in the `BindRegistry`.
    pub fn color(&mut self, color: impl Into<Bind<[f32; 4]>>) -> &mut Self {
        let bind = color.into();
        let initial = bind.read();
        let initial_version = bind.version();
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.color = initial;
        }
        let is_reactive = !matches!(bind, Bind::Value(_));
        let is_animated = bind.animation().is_some();
        if is_reactive || is_animated {
            let displayed = if is_animated {
                Some(Signal::new(initial))
            } else {
                None
            };
            self.ctx.binds.color.push(ColorBindSlot {
                node_id: self.id,
                bind,
                last_version: initial_version,
                displayed,
            });
        }
        self
    }

    pub fn rgb(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.color([r, g, b, 1.0])
    }

    pub fn rgba(&mut self, r: f32, g: f32, b: f32, a: f32) -> &mut Self {
        self.color([r, g, b, a])
    }

    pub fn radius(&mut self, r: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_radius = [r; 4];
        }
        self
    }

    pub fn radii(&mut self, tl: f32, tr: f32, bl: f32, br: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_radius = [tl, tr, bl, br];
        }
        self
    }

    pub fn border(&mut self, width: f32, color: [f32; 4]) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.border_width = width;
            n.style.border_color = color;
        }
        self
    }

    pub fn shadow(
        &mut self,
        offset: [f32; 2],
        blur: f32,
        color: [f32; 4],
        opacity: f32,
    ) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.shadow_offset = offset;
            n.style.shadow_blur = blur;
            n.style.shadow_color = color;
            n.style.shadow_opacity = opacity;
        }
        self
    }

    pub fn opacity(&mut self, o: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.opacity = o;
        }
        self
    }

    pub fn hidden(&mut self) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.visible = false;
        }
        self
    }

    pub fn roughness(&mut self, r: f32) -> &mut Self {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            n.style.roughness = r;
        }
        self
    }

    pub fn on_hover(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.hover = Some(signal));
        self
    }

    pub fn on_press(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.pressed = Some(signal));
        self
    }

    pub fn on_focus(&mut self, signal: Signal<bool>) -> &mut Self {
        self.with_interact(|i| i.focused = Some(signal));
        self
    }

    fn with_interact(&mut self, f: impl FnOnce(&mut NodeInteract)) {
        if let Some(n) = self.ctx.tree.get_mut_raw(self.id) {
            f(&mut n.interact);
        }
    }

    /// Open a nested scope rooted at this node. The closure receives
    /// a `Scene` whose `parent` is set to this node, so any rect/glass
    /// it creates becomes a child here.
    pub fn child<F: FnOnce(&mut Scene)>(&mut self, f: F) -> &mut Self {
        let mut sub = Scene {
            ctx: &mut *self.ctx,
            parent: Some(self.id),
        };
        f(&mut sub);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reactive::{animated, Computed};
    use crate::Curve;
    use std::time::Duration;

    #[test]
    fn nested_children_register_under_parent() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("root")
                .pos(0.0, 0.0)
                .size(100.0, 100.0)
                .rgba(1.0, 0.0, 0.0, 1.0)
                .child(|p| {
                    p.rect("a").pos(10.0, 10.0).size(20.0, 20.0);
                    p.rect("b").pos(40.0, 10.0).size(20.0, 20.0);
                });
        }
        let root_id = ctx.node("root").unwrap();
        let a_id = ctx.node("a").unwrap();
        let b_id = ctx.node("b").unwrap();
        let root = ctx.tree.get(root_id).unwrap();
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0], a_id);
        assert_eq!(root.children[1], b_id);
        assert_eq!(ctx.tree.len(), 3);
    }

    #[test]
    fn raw_color_does_not_register_bind() {
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size(10.0, 10.0).rgba(0.5, 0.5, 0.5, 1.0);
        }
        assert!(ctx.binds.color.is_empty());
    }

    #[test]
    fn signal_color_registers_bind() {
        let s = Signal::new([1.0_f32, 0.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size(10.0, 10.0).color(s.clone());
        }
        assert_eq!(ctx.binds.color.len(), 1);
        let slot = &ctx.binds.color[0];
        assert!(slot.displayed.is_none());
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn computed_color_registers_bind_with_initial_value() {
        let lit = Signal::new(false);
        let c = Computed::new((lit.clone(),), |(l,)| {
            if l { [0.0, 1.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0, 1.0] }
        });
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene.rect("a").size(10.0, 10.0).color(c);
        }
        assert_eq!(ctx.binds.color.len(), 1);
        let slot = &ctx.binds.color[0];
        let n = ctx.tree.get(slot.node_id).unwrap();
        assert_eq!(n.style.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn animated_color_allocates_displayed_signal() {
        let s = Signal::new([0.0_f32, 0.0, 0.0, 1.0]);
        let mut ctx = SceneCtx::new();
        {
            let mut scene = Scene::root(&mut ctx);
            scene
                .rect("a")
                .size(10.0, 10.0)
                .color(animated(s.clone(), Curve::EaseInOut, Duration::from_millis(220)));
        }
        let slot = &ctx.binds.color[0];
        assert!(slot.displayed.is_some());
        assert_eq!(slot.displayed.as_ref().unwrap().get(), [0.0, 0.0, 0.0, 1.0]);
    }
}
