//! 2D and 3D rendering
//!
//! The easiest way to render something is with
//! [`RenderConfig::run`](RenderConfig::run); you can also use the lower-level
//! functions ([`render2d`](render2d()) and [`render3d`](render3d())) for manual
//! control over the input tape.
mod config;
mod render2d;
mod render3d;

pub use config::RenderConfig;
pub use render2d::render as render2d;
pub use render3d::render as render3d;

pub use render2d::{BitRenderMode, DebugRenderMode, RenderMode};
