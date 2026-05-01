//! Reusable pointer-input bookkeeping.
//!
//! `InputState` owns the cursor/hover/capture/focus state and knows how
//! to sync it into the `Signal<bool>` slots on each interactive node.
//! It does **not** touch winit directly — callers translate their event
//! source into the four entry points (`on_cursor_moved`, `on_cursor_left`,
//! `on_left_pressed`, `on_left_released`) and pass the hit-test cache +
//! node tree alongside.
//!
//! The hit-test cache must be produced by `NodeTree::flatten` (topmost
//! first) and rebuilt whenever `TRANSFORM` or `TREE` dirty bits fire.

use crate::node::{HitEntry, NodeId, NodeTree};

/// Transient result returned by each event method so the caller can
/// decide whether to re-flush the tree + request a redraw.
#[derive(Default, Debug, Clone, Copy)]
pub struct InputChange {
    pub hovered_changed: bool,
    pub pressed_changed: bool,
    pub focused_changed: bool,
}

impl InputChange {
    pub fn any(&self) -> bool {
        self.hovered_changed || self.pressed_changed || self.focused_changed
    }
}

#[derive(Default, Debug, Clone)]
pub struct InputState {
    /// Last known cursor position in physical pixels.
    pub cursor: Option<[f32; 2]>,
    /// Topmost interactive node under the cursor right now. While
    /// `captured` is set, this is pinned to the captured node regardless
    /// of where the cursor actually is (matches native button feel).
    pub hovered: Option<NodeId>,
    /// Node that received a press and owns pointer capture until release.
    pub captured: Option<NodeId>,
    /// Most recently focused node (last clicked one that wanted focus).
    pub focused: Option<NodeId>,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk the hit cache top-down and return the first interactive node
    /// whose AABB contains the point. O(n) over interactive nodes only —
    /// `hits` is already filtered during flatten.
    pub fn hit_test(hits: &[HitEntry], x: f32, y: f32) -> Option<NodeId> {
        hits.iter().find(|h| h.contains(x, y)).map(|h| h.node_id)
    }

    /// Process a cursor move. Updates `hovered` (or keeps it pinned when
    /// captured), and while captured also updates `pressed` signals so
    /// dragging off the captured node visually un-presses it.
    pub fn on_cursor_moved(
        &mut self,
        x: f32,
        y: f32,
        hits: &[HitEntry],
        tree: &NodeTree,
    ) -> InputChange {
        self.cursor = Some([x, y]);

        let new_hover = if self.captured.is_some() {
            self.captured
        } else {
            Self::hit_test(hits, x, y)
        };

        let mut change = InputChange::default();
        if new_hover != self.hovered {
            self.hovered = new_hover;
            change.hovered_changed = sync_bool_signals(hits, tree, self.hovered, |n| {
                &n.interact.hover
            });
        }

        if let Some(cap) = self.captured {
            let over = hits
                .iter()
                .find(|h| h.node_id == cap)
                .map(|h| h.contains(x, y))
                .unwrap_or(false);
            let pressed_target = if over { Some(cap) } else { None };
            change.pressed_changed =
                sync_bool_signals(hits, tree, pressed_target, |n| &n.interact.pressed);
        }

        change
    }

    /// Clear hover on cursor leave.
    pub fn on_cursor_left(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        self.cursor = None;
        let mut change = InputChange::default();
        if self.hovered.is_some() {
            self.hovered = None;
            change.hovered_changed =
                sync_bool_signals(hits, tree, None, |n| &n.interact.hover);
        }
        change
    }

    /// Left-button press: capture currently hovered, set pressed + focused.
    pub fn on_left_pressed(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        let target = self.hovered;
        self.captured = target;
        let mut change = InputChange::default();
        change.pressed_changed =
            sync_bool_signals(hits, tree, target, |n| &n.interact.pressed);
        if self.focused != target {
            self.focused = target;
            change.focused_changed =
                sync_bool_signals(hits, tree, target, |n| &n.interact.focused);
        }
        change
    }

    /// Left-button release: clear pressed state and re-evaluate hover at
    /// the current cursor position.
    pub fn on_left_released(&mut self, hits: &[HitEntry], tree: &NodeTree) -> InputChange {
        self.captured = None;
        let mut change = InputChange::default();
        change.pressed_changed = sync_bool_signals(hits, tree, None, |n| &n.interact.pressed);
        if let Some([x, y]) = self.cursor {
            let new_hover = Self::hit_test(hits, x, y);
            if new_hover != self.hovered {
                self.hovered = new_hover;
                change.hovered_changed =
                    sync_bool_signals(hits, tree, self.hovered, |n| &n.interact.hover);
            }
        }
        change
    }
}

/// Iterate the hit cache once and write `target == node_id` into the
/// signal returned by `select` for each interactive node. Returns true
/// if any signal flipped. `Signal::set` is a no-op write if unchanged.
fn sync_bool_signals(
    hits: &[HitEntry],
    tree: &NodeTree,
    target: Option<NodeId>,
    select: impl Fn(&crate::node::Node) -> &Option<crate::signal::Signal<bool>>,
) -> bool {
    let mut changed = false;
    for entry in hits {
        if let Some(n) = tree.get(entry.node_id)
            && let Some(sig) = select(n).as_ref() {
                let on = Some(entry.node_id) == target;
                if sig.set(on) {
                    changed = true;
                }
            }
    }
    changed
}
