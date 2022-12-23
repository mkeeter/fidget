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
//! let sum = ctx.add(x, y).unwrap();
//! ```
//!
//! As you can see, this is very verbose.  As an alternative, Fidget includes
//! bindings to [Rhai](https://rhai.rs), a simple Rust-native scripting
//! language, in the [`fidget::rhai` namespace](crate::rhai):
//!
//! ```
//! use fidget::rhai::eval;
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! ```
//!
//! # Evaluation
//! The main operation performed on an implicit surface is **evaluation**, i.e.
//! passing it some position `(x, y, z)` and getting back a result.  This will
//! be done _a lot_, so it has to be fast.
//!
//! Before evaluation, a shape must be baked into a [`Tape`](crate::eval::Tape).
//! This is performed by [`Context::get_tape`](crate::Context::get_tape):
//! ```
//! use fidget::{rhai::eval, vm};
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! let tape = ctx.get_tape::<vm::Eval>(sum);
//! assert_eq!(tape.len(), 3); // X, Y, and (X + Y)
//! ```
//!
//! A tape is set of operations for a very simple virtual machine; the
//! expression above would be something like
//! ```text
//! $0 = INPUT 0   // X
//! $1 = INPUT 1   // Y
//! $2 = ADD $0 $1 // (X + Y)
//! ```
//!
//! To evaluate the tape, we must choose an [evaluator
//! family](crate::eval::Family).  At the moment, Fidget implements two
//! evaluator families:
//!
//! - [`fidget::jit::Eval`](crate::jit::Eval) performs fast evaluation by
//!   compiling shapes down to native code.  This is only functional on an ARM64
//!   system running natively.
//! - [`fidget::vm::Eval`](crate::vm::Eval) evaluates
//!   using an interpreter.  This is slower, but can run in more situations (e.g.
//!   x86 machines or in WebAssembly).
//!
//! Looking at the [`eval::Family`](crate::eval::Family) trait, you may notice
//! that it requires several different kinds of evaluation.  In addition to
//! supporting evaluation at a single point, Fidget evaluators must support
//!
//! - Evaluation on a array of points (giving an opportunity for SIMD)
//! - Interval evaluation
//! - Evaluation of partial derivatives with respect to `x, y, z`
//!
//! These evaluation flavors are used in rendering:
//! - SIMD evaluation speeds up rendering groups of voxels.
//! - Interval evaluation can prove large regions of space to be empty or full,
//!   at which point they don't need to be considered further.
//! - At the surface of the model, partial derivatives represent normals and
//!   can be used for shading.
//!
//! Here's a simple example of interval evaluation:
//! ```
//! use fidget::{eval::Eval, rhai::eval, vm};
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! let tape = ctx.get_tape(sum);
//! let mut interval_eval = vm::Eval::new_interval_evaluator(tape);
//! let (out, _) = interval_eval.eval_i(
//!         [0.0, 1.0], // X
//!         [2.0, 3.0], // Y
//!         [0.0, 0.0], // Z
//!         &[]         // variables (unused)
//!     ).unwrap();
//! assert_eq!(out, [2.0, 4.0].into());
//! ```
//!
//! # Tape simplification
//! Interval evaluation serves two purposes.  As we already mentioned, it can be
//! used to prove large regions empty or filled, which lets us do less work when
//! rendering.  In addition, it can discover **sections of the tape** that are
//! always inactive in a particular spatial region.
//!
//! Consider evaluating `f(x, y, z) = max(x, y)` with `x = [0, 1]` and
//! `y = [2, 3]`:
//! ```
//! use fidget::{eval::Eval, rhai::eval, vm};
//!
//! let (sum, ctx) = eval("min(x, y)").unwrap();
//! let tape = ctx.get_tape(sum);
//! let mut interval_eval = vm::Eval::new_interval_evaluator(tape);
//! let (out, _) = interval_eval.eval_i(
//!         [0.0, 1.0], // X
//!         [2.0, 3.0], // Y
//!         [0.0, 0.0], // Z
//!         &[]         // variables (unused)
//!     ).unwrap();
//! assert_eq!(out, [0.0, 1.0].into());
//! ```
//!
//! In the evaluation region `x = [0, 1]; y = [2, 3]`, `x` is **strictly less
//! than** `y` in the `min(x, y)` clause.  This means that we can simplify the
//! tape from `f(x, y, z) = min(x, y) → f(x, y, z) = x`.
//!
//! Simplification is done with
//! [`IntervalEval::simplify`](crate::eval::IntervalEval::simplify).
//! This is _stateful_: `simplify` uses the most recent evaluation to decide how
//! to simplify the tape.
//!
//! ```
//! # use fidget::{eval::Eval, rhai::eval, vm};
//! # let (sum, ctx) = eval("min(x, y)").unwrap();
//! # let tape = ctx.get_tape(sum);
//! # let mut interval_eval = vm::Eval::new_interval_evaluator(tape);
//! # let (out, _) = interval_eval.eval_i(
//! #         [0.0, 1.0], // X
//! #         [2.0, 3.0], // Y
//! #         [0.0, 0.0], // Z
//! #         &[]         // variables (unused)
//! #     ).unwrap();
//! // (same code as above)
//! assert_eq!(interval_eval.tape().len(), 3);
//! let new_tape = interval_eval.simplify();
//! assert_eq!(new_tape.len(), 1); // just the 'X' term
//! ```
//!
//! Remember that this simplified tape is only valid for points (or intervals)
//! within the interval region `x = [0, 1]; y = [2, 3]`.  It's up to you to make
//! sure this is upheld!
//!
//! # Rasterization
//! At the moment, Fidget uses all of this machinery to build one user-facing
//! algorithm: rasterization of implicit surfaces.
//!
//! (two, if you count 2D and 3D rasterization separately)
//!
//! This is implemented in the [`fidget::render` namespace](crate::render).
//!
//! Here's a quick example:
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
//!
//! # Similar projects
//! Fidget overlaps with various projects in the implicit modeling space:
//!
//! - [Antimony: CAD from a parallel universe](https://mattkeeter.com/projects/antimony)*
//! - [`libfive`: Infrastructure for solid modeling](https://libfive.com)*
//! - [Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces (MPR)](https://github.com/mkeeter/mpr)*
//! - [ImplicitCAD: Powerful, Open-Source, Programmatic CAD](https://implicitcad.org/)
//! - [Ruckus: Procedural CAD For Weirdos](https://docs.racket-lang.org/ruckus/index.html)
//! - [Curv: a language for making art using mathematics](https://github.com/curv3d/curv)
//! - [sdf: Simple SDF mesh generation in Python](https://github.com/fogleman/sdf)
//! - Probably more; PRs welcome!
//!
//! *written by the same author
//!
//! Compared to these projects, Fidget is unique in having a native JIT and
//! using that JIT while performing tape simplification.  This makes it _blazing
//! fast_.
//! For example, here are rough benchmarks rasterizing [this model](https://www.mattkeeter.com/projects/siggraph/depth_norm@2x.png)
//! across three different implementations:
//!
//! Size | `libfive` | MPR| Fidget
//! -|-|-|-
//! 1024³ | 66.8 ms | 22.6 ms| 23.6 ms
//! 1536³ | 127 ms | 39.3 ms| 45.4 ms
//! 2048³ | 211 ms | 60.6 ms| 77.4 ms
//!
//! `libfive` and Fidget are running on an M1 Max CPU; MPR is running on a GTX
//! 1080 Ti GPU.
//!
//! Fidget is missing a bunch of features that are found in more mature
//! projects.  For example, it does not include mesh export, and only includes a
//! debug GUI.

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
