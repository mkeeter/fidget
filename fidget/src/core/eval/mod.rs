//! Traits and data structures for evaluation
//!
//! The easiest way to build an evaluator of a particular kind is by calling
//! `new_X_evaluator` on [`Tape`].
//!
//! ```rust
//! use fidget::vm;
//! use fidget::context::Context;
//! use fidget::eval::Tape;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let tape = Tape::<vm::Eval>::new(&ctx, x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = tape.new_point_evaluator();
//! assert_eq!(eval.eval(0.25, 0.0, 0.0, &[])?.0, 0.25);
//! # Ok::<(), fidget::Error>(())
//! ```

// Evaluators
pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;

pub mod bulk;
pub mod tape;
pub mod tracing;
pub mod types;

mod vars;

// Re-export a few things
pub use tape::TapeData;
pub use tracing::Choice;
pub use vars::Vars;

use bulk::BulkEvaluator;
use tracing::TracingEvaluator;
use types::{Grad, Interval};

/// Helper trait to enforce equality of two types
pub trait TyEq {}
impl<T> TyEq for (T, T) {}

/// A shape can produce a float slice evaluator
pub trait ShapeFloatSliceEval {
    /// Bulk point evaluator
    type Eval: BulkEvaluator<f32>;

    /// Returns a tape for use in a float slice evaluator
    fn tape(&self) -> <Self::Eval as BulkEvaluator<f32>>::Tape;
}

/// A shape can produce a grad slice evaluator
pub trait ShapeGradSliceEval {
    /// Bulk grad evaluator
    type Eval: BulkEvaluator<Grad>;

    /// Returns a tape for use in a float slice evaluator
    fn tape(&self) -> <Self::Eval as BulkEvaluator<Grad>>::Tape;
}

/// A shape can produce an interval evaluator
pub trait ShapeIntervalEval {
    /// Interval evaluator
    type Eval: TracingEvaluator<types::Interval>;

    /// Returns a tape for use in a single interval tracing evaluator
    fn tape(&self) -> <Self::Eval as TracingEvaluator<Interval>>::Tape;
}

/// A shape can produce a point evaluator
pub trait ShapePointEval {
    /// Point evaluator
    type Eval: TracingEvaluator<f32>;

    /// Returns a tape for use in a single interval tracing evaluator
    fn tape(&self) -> <Self::Eval as TracingEvaluator<f32>>::Tape;
}

/// A shape can be simplified
pub trait ShapeSimplify {
    /// Associated types for traces captured during evaluation
    type Trace;

    /// Generates a simplified shape based on a captured trace
    ///
    /// # Panics
    /// This function may panic if the trace is incompatible with this shape
    fn simplify(&self, trace: &Self::Trace) -> Self;
}

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented; we simply
/// require that the shape can generate evaluators of various kinds.
///
/// This trait doesn't actually implement any functions itself; it simply
/// stitches together a bunch of other traits with appropriate equality
/// constraints.
pub trait Shape:
    ShapeFloatSliceEval
    + ShapeGradSliceEval
    + ShapePointEval
    + ShapeIntervalEval
    + ShapeSimplify
    + ShapeRenderHints
where
    (
        <<Self as ShapePointEval>::Eval as TracingEvaluator<f32>>::Trace,
        Self::Trace,
    ): TyEq,
    (
        <<Self as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace,
        Self::Trace,
    ): TyEq,
{
    // Nothing to add here
}

/// A shape can offer hints as to how it should be rendered
pub trait ShapeRenderHints {
    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];

    /// Indicates whether we run tape simplification at the given cell depth
    /// during meshing.
    ///
    /// By default, this is always true; for evaluators where simplification is
    /// more expensive than evaluation (i.e. the JIT), it may only be true at
    /// certain depths.
    fn simplify_tree_during_meshing(_d: usize) -> bool {
        true
    }
}

/// A [`Shape`] which is generated from a math tree
pub trait MathShape {
    /// Build a new shape from the given math tree
    fn new(ctx: &crate::context::Context, node: crate::context::Node) -> Self;
}
