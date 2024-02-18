//! Fidget is a library of infrastructure and algorithms for complex closed-form
//! implicit surfaces.
//!
//! An **implicit surface** is a function `f(x, y, z)`, where `x`, `y`, and `z`
//! represent a position in 3D space.  By convention, if `f(x, y, z) < 0`, then
//! that position is _inside_ the shape; if it's `> 0`, then that position is
//! _outside_ the shape; otherwise, it's on the boundary of the shape.
//!
//! A **closed-form** implicit surface means that the function is given as a
//! fixed expression built from closed-form operations (addition, subtraction,
//! etc), with no mutable state.  This is in contrast to
//! [ShaderToy](https://www.shadertoy.com/)-style implicit surface functions,
//! which often include mutable state and make control-flow decisions at
//! runtime.
//!
//! Finally, **complex** means that that the library scales to expressions with
//! thousands of clauses.
//!
//! Details on overall project status are in the
//! [project's README](https://github.com/mkeeter/fidget);
//! the rest of this page is a quick tour through the library APIs.
//!
//! # Shape construction
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
//! let sum = ctx.add(x, y)?;
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! As an alternative, Fidget includes bindings to [Rhai](https://rhai.rs), a
//! simple Rust-native scripting language, in the [`fidget::rhai`
//! namespace](crate::rhai).  These bindings offer a terser way to construct
//! a shape from a script:
//!
//! ```
//! use fidget::rhai;
//!
//! let (sum, ctx) = rhai::eval("x + y")?;
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! # Evaluation
//! The main operation performed on an implicit surface is **evaluation**, i.e.
//! passing it some position `(x, y, z)` and getting back a result.  This will
//! be done _a lot_, so it has to be fast.
//!
//! Evaluation is deliberately agnostic to the specific details of how we go
//! from position to results.  This abstraction is represented by the
//! [`Shape` trait](crate::eval::Shape), which defines how to make both
//! **evaluators** and **tapes**.
//!
//! An **evaluator** is an object which performs evaluation of some kind (point,
//! array, gradient, interval).  It carries no persistent data, and would
//! typically be constructed on a per-thread basis.
//!
//! A **tape** contains instructions for an evaluator.
//!
//! For example, the [`VmShape`](crate::vm::VmShape) type uses a simple virtual
//! machine to perform evaluation.  Here's a peek at the internals:
//!
//! At the moment, Fidget implements two kinds of shapes:
//!
//! - [`fidget::vm::VmShape`](crate::vm::VmShape) evaluates a list of opcodes
//!   using an interpreter.  This is slower, but can run in more situations (e.g.
//!   in WebAssembly).
//! - [`fidget::jit::JitShape`](crate::jit::JitShape) performs fast evaluation
//!   by compiling shapes down to native code.
//!
//! Looking at the [`eval::Shape`](crate::eval::Shape) trait, you may notice
//! that it requires four different kinds of evaluation:
//!
//! - Single-point evaluation
//! - Interval evaluation
//! - Evaluation on an array of points, returning `f32` values
//! - Evaluation on an array of points, returning partial derivatives with
//!   respect to `x, y, z`
//!
//! These evaluation flavors are used in rendering:
//! - Interval evaluation can conservatively prove large regions of space to be
//!   empty or full, at which point they don't need to be considered further.
//! - Array-of-points evaluation speeds up calculating occupancy (inside /
//!   outside) when given a set of voxels, because dispatch overhead is
//!   amortized over many points.
//! - At the surface of the model, partial derivatives represent normals and
//!   can be used for shading.
//!
//! Here's a simple example of interval evaluation:
//! ```
//! use fidget::{eval::{Shape, EzShape, TracingEvaluator}, rhai, vm::VmShape};
//!
//! let (sum, ctx) = rhai::eval("x + y")?;
//! let shape = VmShape::new(&ctx, sum)?;
//! let mut interval_eval = VmShape::new_interval_eval();
//! let tape = shape.ez_interval_tape();
//! let (out, _trace) = interval_eval.eval(
//!     &tape,
//!     [0.0, 1.0], // X
//!     [2.0, 3.0], // Y
//!     [0.0, 0.0], // Z
//!     &[]         // variables (unused)
//! )?;
//! assert_eq!(out, [2.0, 4.0].into());
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! # Shape simplification
//! Interval evaluation serves two purposes.  As we already mentioned, it can be
//! used to prove large regions empty or filled, which lets us do less work when
//! rendering.  In addition, it can discover **sections of the tape** that are
//! always inactive in a particular spatial region.
//!
//! Consider evaluating `f(x, y, z) = max(x, y)` with `x = [0, 1]` and
//! `y = [2, 3]`:
//! ```
//! use fidget::{eval::{TracingEvaluator, Shape, EzShape}, rhai, vm::VmShape};
//!
//! let (sum, ctx) = rhai::eval("min(x, y)")?;
//! let shape = VmShape::new(&ctx, sum)?;
//! let mut interval_eval = VmShape::new_interval_eval();
//! let tape = shape.ez_interval_tape();
//! let (out, trace) = interval_eval.eval(
//!     &tape,
//!     [0.0, 1.0], // X
//!     [2.0, 3.0], // Y
//!     [0.0, 0.0], // Z
//!     &[]         // variables (unused)
//! )?;
//! assert_eq!(out, [0.0, 1.0].into());
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! In the evaluation region `x = [0, 1]; y = [2, 3]`, `x` is **strictly less
//! than** `y` in the `min(x, y)` clause.  This means that we can simplify the
//! tape from `f(x, y, z) = min(x, y) → f(x, y, z) = x`.
//!
//! Interval evaluation is a kind of
//! [tracing evaluation](crate::eval::TracingEvaluator), which returns a tuple
//! of `(value, trace)`.  The trace can be used to simplify the original shape:
//!
//! ```
//! # use fidget::{eval::{TracingEvaluator, Shape, EzShape}, rhai, vm::VmShape};
//! # let (sum, ctx) = rhai::eval("min(x, y)")?;
//! # let shape = VmShape::new(&ctx, sum)?;
//! # let mut interval_eval = VmShape::new_interval_eval();
//! # let tape = shape.ez_interval_tape();
//! # let (out, trace) = interval_eval.eval(
//! #         &tape,
//! #         [0.0, 1.0], // X
//! #         [2.0, 3.0], // Y
//! #         [0.0, 0.0], // Z
//! #         &[]         // variables (unused)
//! #     )?;
//! // (same code as above)
//! assert_eq!(tape.size(), 3);
//! let new_shape = shape.ez_simplify(trace.unwrap())?;
//! assert_eq!(new_shape.ez_interval_tape().size(), 1); // just the 'X' term
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! Remember that this simplified tape is only valid for points (or intervals)
//! within the interval region `x = [0, 1]; y = [2, 3]`.  It's up to you to make
//! sure this is upheld!
//!
//! # Rasterization
//! At the moment, Fidget uses all of this machinery to build two user-facing
//! algorithms: rasterization of implicit surfaces in 2D and 3D.
//!
//! They are implemented in the [`fidget::render` namespace](crate::render).
//!
//! Here's a quick example:
//! ```
//! use fidget::context::Context;
//! use fidget::rhai;
//! use fidget::vm::VmShape;
//! use fidget::render::{BitRenderMode, RenderConfig};
//!
//! let (shape, ctx) = rhai::eval("sqrt(x*x + y*y) - 1")?;
//! let cfg = RenderConfig::<2> {
//!     image_size: 32,
//!     ..RenderConfig::default()
//! };
//! let shape = VmShape::new(&ctx, shape)?;
//! let out = cfg.run(shape, &BitRenderMode)?;
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
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! # Feature flags
#![doc = document_features::document_features!()]
#![warn(missing_docs)]

// Re-export everything from fidget::core into the top-level namespace
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

#[cfg(feature = "mesh")]
pub mod mesh;
