//! Fidget is a library of infrastructure and algorithms for function
//! evaluation, with an emphasis on complex closed-form implicit surfaces.
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
//! This is efficient, but is awkward to write.  It's also possible to construct
//! shapes without a [`Context`] using the [`Tree`](crate::context::Tree) type,
//! then import the tree into a context:
//! ```
//! use fidget::context::{Context, Tree};
//!
//! let t = Tree::x() + Tree::y();
//! let mut ctx = Context::new();
//! let sum = ctx.import(&t);
//! ```
//!
//! As a third alternative, Fidget includes bindings to [Rhai](https://rhai.rs),
//! a simple Rust-native scripting language, in the [`fidget::rhai`
//! namespace](crate::rhai).  These bindings allow shapes to be constructed from
//! a script, adding flexibility:
//!
//! ```
//! # use fidget::context::Context;
//! let t = fidget::rhai::eval("x + y")?;
//! let mut ctx = Context::new();
//! let sum = ctx.import(&t);
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
//! [`Function` trait](crate::eval::Function), which defines how to make both
//! **evaluators** and **tapes**.
//!
//! An **evaluator** is an object which performs evaluation of some kind (point,
//! array, gradient, interval).  It carries no persistent data, and would
//! typically be constructed on a per-thread basis.
//!
//! A **tape** contains instructions for an evaluator.
//!
//! At the moment, Fidget implements two kinds of functions:
//!
//! - [`fidget::vm::VmFunction`](crate::vm::VmFunction) evaluates a list of
//!   opcodes using an interpreter.  This is slower, but can run in more
//!   situations (e.g. in WebAssembly).
//! - [`fidget::jit::JitFunction`](crate::jit::JitFunction) performs fast
//!   evaluation by compiling expressions down to native code.
//!
//! The [`Function`](crate::eval::Function) trait requires four different kinds
//! of evaluation:
//!
//! - Single-point evaluation
//! - Interval evaluation
//! - Evaluation on an array of points, returning `f32` values
//! - Evaluation on an array of points, returning partial derivatives with
//!   respect to input variables
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
//! # Functions and shapes
//! The [`Function`](crate::eval::Function) trait supports arbitrary numbers of
//! varibles; when using it for implicit surfaces, it's common to wrap it in a
//! [`Shape`](crate::shape::Shape), which binds `(x, y, z)` axes to specific
//! variables.
//!
//! Here's a simple example of interval evaluation, using a `Shape` to wrap a
//! function and evaluate it at a particular `(x, y, z)` position:
//!
//! ```
//! use fidget::{
//!     context::Tree,
//!     shape::{Shape, EzShape},
//!     vm::VmShape
//! };
//!
//! let tree = Tree::x() + Tree::y();
//! let shape = VmShape::from(tree);
//! let mut interval_eval = VmShape::new_interval_eval();
//! let tape = shape.ez_interval_tape();
//! let (out, _trace) = interval_eval.eval(
//!     &tape,
//!     [0.0, 1.0], // X
//!     [2.0, 3.0], // Y
//!     [0.0, 0.0], // Z
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
//! use fidget::{
//!     context::Tree,
//!     shape::EzShape,
//!     vm::VmShape
//! };
//!
//! let tree = Tree::x().min(Tree::y());
//! let shape = VmShape::from(tree);
//! let mut interval_eval = VmShape::new_interval_eval();
//! let tape = shape.ez_interval_tape();
//! let (out, trace) = interval_eval.eval(
//!     &tape,
//!     [0.0, 1.0], // X
//!     [2.0, 3.0], // Y
//!     [0.0, 0.0], // Z
//! )?;
//! assert_eq!(out, [0.0, 1.0].into());
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! In the evaluation region `x = [0, 1]; y = [2, 3]`, `x` is **strictly less
//! than** `y` in the `min(x, y)` clause.  This means that we can simplify the
//! tape from `min(x, y) →  x`.
//!
//! Interval evaluation is a kind of
//! [tracing evaluation](crate::eval::TracingEvaluator), which returns a tuple
//! of `(value, trace)`.  The trace can be used to simplify the original shape:
//!
//! ```
//! # use fidget::{
//! #     context::Tree,
//! #     shape::EzShape,
//! #     vm::VmShape
//! # };
//! # let tree = Tree::x().min(Tree::y());
//! # let shape = VmShape::from(tree);
//! assert_eq!(shape.size(), 3); // min, X, Y
//! # let mut interval_eval = VmShape::new_interval_eval();
//! # let tape = shape.ez_interval_tape();
//! # let (out, trace) = interval_eval.eval(
//! #         &tape,
//! #         [0.0, 1.0], // X
//! #         [2.0, 3.0], // Y
//! #         [0.0, 0.0], // Z
//! #     )?;
//! // (same code as above)
//! let new_shape = shape.ez_simplify(trace.unwrap())?;
//! assert_eq!(new_shape.size(), 1); // just the X term
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! Remember that this simplified tape is only valid for points (or intervals)
//! within the interval region `x = [0, 1]; y = [2, 3]`.  It's up to you to make
//! sure this is upheld!
//!
//! # Rasterization
//! Fidget implements both 2D and 3D rasterization of implicit surfaces,
//! implemented in the [`fidget::render` module](render).
//!
//! Here's a quick example:
//! ```
//! use fidget::{
//!     context::{Tree, Context},
//!     render::{BitRenderMode, RenderConfig},
//!     vm::VmShape,
//! };
//!
//! let x = Tree::x();
//! let y = Tree::y();
//! let tree = (x.square() + y.square()).sqrt() - 1.0;
//! let cfg = RenderConfig::<2> {
//!     image_size: 32,
//!     ..RenderConfig::default()
//! };
//! let shape = VmShape::from(tree);
//! let out = cfg.run::<_, BitRenderMode>(shape)?;
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
//! # Meshing
//! Fidget implements
//! [Manifold Dual Contouring](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf),
//! which converts from implicit surfaces to triangle meshes.
//!
//! This is documented in the [`fidget::mesh`](mesh) module.
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

#[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
pub mod jit;

#[cfg(feature = "mesh")]
pub mod mesh;
