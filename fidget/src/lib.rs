//! Infrastructure and algorithms for complex closed-form implicit surfaces.
//!
//! ```
//! use fidget::context::Context;
//! use fidget::rhai::eval;
//! use fidget::render::{render2d::render, config::RenderConfig};
//!
//! let (shape, ctx) = eval("sqrt(x*x + y*y) - 1").unwrap();
//! let tape = ctx.get_tape(shape, u8::MAX);
//! let cfg = RenderConfig::<2> {
//!     image_size: 32,
//!     ..RenderConfig::default()
//! };
//! // TODO: actually render stuff here
//!
//! // This will print
//! //           XXXXXXXXXX
//! //       XXXXXXXXXXXXXXXXXX
//! //     XXXXXXXXXXXXXXXXXXXXXX
//! //   XXXXXXXXXXXXXXXXXXXXXXXXXX
//! //   XXXXXXXXXXXXXXXXXXXXXXXXXX
//! // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//! // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//! // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//! // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//! // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
//! //   XXXXXXXXXXXXXXXXXXXXXXXXXX
//! //   XXXXXXXXXXXXXXXXXXXXXXXXXX
//! //     XXXXXXXXXXXXXXXXXXXXXX
//! //       XXXXXXXXXXXXXXXXXX
//! //           XXXXXXXXXX
//! ```

// Re-export everything from fidget_core into the top-level namespace
mod core;
pub use crate::core::*;

#[cfg(feature = "render")]
pub mod render;

#[cfg(feature = "rhai")]
pub mod rhai;

#[cfg(feature = "jit")]
pub mod jit;
