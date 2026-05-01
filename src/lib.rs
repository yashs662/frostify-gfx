//! frostify-gfx — reactive GPU UI rendering library.
//!
//! Transparent window, SDF shapes with solid colors and frosted glass
//! (per-instance blur + edge refraction), text via cosmic-text, image
//! atlas blits, and a flex-style layout engine.
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
pub mod layout;
pub mod node;
pub mod reactive;
pub mod scene;
pub mod signal;
pub mod text;

pub use anim::{Curve, Lerp, TickResult, Timeline, Tween};
pub use app::{App, AppConfig, HeadlessHelper};
pub use gpu::{
    FrameStats, FrameTiming, FrameUniform, GpuContext, ImageAtlas, ImageEntry, ImageHandle,
    MemoryReport, ShapeInstance,
};
pub use input::{InputChange, InputState};
pub use layout::{Align, Axis, Justify, LayoutStyle, Len, Measurer, NullMeasurer};
pub use node::{
    dirty, HitEntry, ImageRef, Node, NodeBuilder, NodeId, NodeTree, ShapeKind, ShapeStyle,
    TextRef, WindowAction,
};
pub use reactive::{animated, AnimatedBind, Bind, Computed, DepTuple, Source};
pub use scene::{
    BindRegistry, ColorBindSlot, NodeBuilderRef, PositionBindSlot, Scene, SceneCtx, SizeBindSlot,
};
pub use signal::Signal;
pub use text::{RasterizedGlyph, ShapedGlyph, TextMetrics, TextResources};
