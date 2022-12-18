//! Fidget is a library of infrastructure and algorithms for complex closed-form
//! implicit surfaces.
//!
//! An **implicit surface** is a function `f(x, y, z)`, where `x`, `y`, and `z`
//! represent a position in 3D space.  By convention, if `f(x, y, z) < 0`, then
//! that position is **inside** the shape; if it's `> 0`, then that position is
//! **outside** the shape; otherwise, it's on the boundary of the shape.
//!
//! A **closed-form** implicit surface means that the function is given in terms
//! of closed-form operations (addition, subtraction, etc).  This is in contrast
//! to [ShaderToy](https://www.shadertoy.com/)-style implicit surface functions,
//! which often use explicit looping and recursion.
//!
//! Finally, **complex** means that that the library scales to expressions with
//! thousands of clauses.
//!
//! # Core functionality
//! Fidget implements two core pieces of functionality, then uses them to build
//! additional tools.
//!
//! ## Shape construction
//! Shapes are constructed within a
//! [`fidget::context::Context`](crate::context::Context).  A context serves as
//! an arena-style allocator, doing local deduplication and other simple
//! optimizations (e.g. constant folding).
//!
//! Shapes can be constructed manually, using functions on a context:
//! ```
//! use fidget::context::Context;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let y = ctx.y();
//! let sum = ctx.add(x, y).unwrap();
//! ```
//!
//! This is a very manual process.  As an alternative, Fidget includes bindings
//! to [Rhai](https://rhai.rs), a simple Rust-native scripting language,
//! in the [`fidget::rhai` namespace](crate::rhai):
//!
//! ```
//! use fidget::rhai::eval;
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! ```
//!
//! ## Evaluation
//! Here's where things get interesting!
//!
//! The main operation performed on an implicit surface is **evaluation**, i.e.
//! passing it some position `(x, y, z)` and getting back a result.  Doing so
//! efficiently is the core challenge that Fidget addresses.
//!
//! Before evaluation, a shape must be baked into a [`Tape`](crate::eval::Tape).
//! This is performed by [`Context::get_tape`](crate::Context::get_tape):
//! ```
//! use fidget::{rhai::eval};
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! let tape = ctx.get_tape(sum).unwrap();
//! assert_eq!(tape.len(), 3); // X, Y, and (X + Y)
//! ```
//!
//! To evaluate the tape, we must choose an evaluator family.  At the moment,
//! Fidget implements two evaluator families:
//!
//! - [`fidget::jit::Eval`](crate::jit::Eval) performs fast evaluation by
//!   compiling shapes down to native code.  This is only functional on an ARM64
//!   system running natively.
//! - [`fidget::vm::Eval`](crate::vm::Eval) evaluates
//!   using an interpreter.  This is slower, but can run in more situations (e.g.
//!   x86 machines or in WebAssembly).
//!
//! The naive approach to implicit rendering is to evaluate a the at every point
//! in space.  This is slow, because 3D space grows as `N^3`.
//!
//! A more efficient strategy dates back to [Duff '92](https://dl.acm.org/doi/10.1145/142920.134027):
//! - Evaluate regions of the space using [interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic)
//!     - Skip sections of the space that are unambiguously empty or full
//!     - Construct simplified versions of the shape based on interval results
//! - Subdivide and recurse
//!
//! Fidget includes built-in support for interval arithmetic:
//! ```
//! use fidget::{eval::Eval, rhai::eval, vm};
//! let (sum, ctx) = eval("x + y").unwrap();
//! let tape = ctx.get_tape(sum);
//! let interval_eval = vm::Eval::new_interval_evaluator(tape);
//! ```
//!
//! # How is this different from _X_?
//!
//! # Simple example
//! ```
//! use fidget::context::Context;
//! use fidget::rhai::eval;
//! use fidget::vm;
//! use fidget::render::{render2d::{BitRenderMode}, config::RenderConfig};
//!
//! let (shape, ctx) = eval("sqrt(x*x + y*y) - 1").unwrap();
//! let cfg = RenderConfig::<2> {
//!     image_size: 32,
//!     ..RenderConfig::default()
//! };
//! let out = cfg.run::<BitRenderMode, vm::Eval>(shape, ctx);
//! let mut iter = out.iter();
//! for y in 0..cfg.image_size {
//!     for x in 0..cfg.image_size {
//!         if *iter.next().unwrap() {
//!             print!("XX");
//!         } else {
//!             print!("  ");
//!         }
//!     }
//!     println!();
//! }
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

mod error;
pub use error::Error;

#[cfg(feature = "render")]
pub mod render;

#[cfg(feature = "rhai")]
pub mod rhai;

#[cfg(feature = "jit")]
pub mod jit;
