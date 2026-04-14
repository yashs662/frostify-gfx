//! frostify-gfx — reactive GPU UI rendering library.
//!
//! Stage 1 scope: transparent window, SDF shapes with solid colors and
//! glass/roughness. No text, no images, no layout engine. Absolute pixel
//! coordinates for debug layouts are available.
//!
//! The crate is a **library** — it does not own a window or an event
//! loop. The public surface is:
//!
//! - [`GpuContext`] — wgpu setup, multi-pass renderer, headless capture.
//! - [`NodeTree`], [`Node`], [`NodeId`], [`HitEntry`] — retained scene graph.
//! - [`Signal`] — reactive value primitive used by interactive nodes.
//! - [`InputState`] — cursor/hover/press bookkeeping that consumers plug
//!   into whichever event source they use (e.g. winit).
//! - [`debug`] — PNG screenshot helper for manual + headless verification.
//!
//! See `examples/hello_window.rs` for a full integration that builds a
//! winit event loop, a demo scene, and env-var-driven headless captures.

pub mod anim;
pub mod app;
pub mod debug;
pub mod gpu;
pub mod input;
pub mod node;
pub mod reactive;
pub mod scene;
pub mod signal;

pub use anim::{Curve, Lerp, TickResult, Timeline, Tween};
pub use app::{App, AppConfig, HeadlessHelper};
pub use gpu::{FrameUniform, GpuContext, ShapeInstance};
pub use input::{InputChange, InputState};
pub use node::{dirty, HitEntry, Node, NodeBuilder, NodeId, NodeTree, ShapeKind, ShapeStyle};
pub use reactive::{animated, AnimatedBind, Bind, Computed, DepTuple, Source};
pub use scene::{BindRegistry, ColorBindSlot, NodeBuilderRef, Scene, SceneCtx};
pub use signal::Signal;
