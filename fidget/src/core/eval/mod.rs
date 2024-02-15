//! Traits and data structures for evaluation
//!
//! The easiest way to build an evaluator of a particular kind is by calling
//! `new_X_evaluator` on [`Tape`].
//!
//! ```rust
//! use fidget::vm::VmShape;
//! use fidget::context::Context;
//! use fidget::eval::{TracingEvaluator, Shape};
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let shape = VmShape::new(&ctx, x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.point_tape();
//! let (value, _trace) = eval.eval(&tape, 0.25, 0.0, 0.0, &[])?;
//! assert_eq!(value, 0.25);
//! # Ok::<(), fidget::Error>(())
//! ```
use crate::Error;
use std::{collections::HashMap, sync::Arc};

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
pub use bulk::BulkEvaluator;
pub use tape::TapeData;
pub use tracing::Choice;
pub use tracing::TracingEvaluator;
pub use vars::Vars;

use types::{Grad, Interval};

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented; we simply
/// require that the shape can generate evaluators of various kinds.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
pub trait Shape: Send + Sync + Clone {
    /// Associated type traces collected during tracing evaluation
    type Trace;

    /// Associated type for single-point tracing evaluation
    type PointEval: TracingEvaluator<f32, Trace = Self::Trace> + Send + Sync;

    /// Builds a new point evaluator
    fn new_point_eval() -> Self::PointEval {
        Self::PointEval::new()
    }

    /// Associated type for single interval tracing evaluation
    type IntervalEval: TracingEvaluator<Interval, Trace = Self::Trace>
        + Send
        + Sync;

    /// Builds a new interval evaluator
    fn new_interval_eval() -> Self::IntervalEval {
        Self::IntervalEval::new()
    }

    /// Associated type for evaluating many points in one call
    type FloatSliceEval: BulkEvaluator<f32> + Send + Sync;

    /// Builds a new float slice evaluator
    fn new_float_slice_eval() -> Self::FloatSliceEval {
        Self::FloatSliceEval::new()
    }

    /// Associated type for evaluating many gradients in one call
    type GradSliceEval: BulkEvaluator<Grad> + Send + Sync;

    /// Builds a new gradient slice evaluator
    fn new_grad_slice_eval() -> Self::GradSliceEval {
        Self::GradSliceEval::new()
    }

    /// Returns an evaluation tape for a point evaluator
    fn point_tape(&self) -> <Self::PointEval as TracingEvaluator<f32>>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn interval_tape(
        &self,
    ) -> <Self::IntervalEval as TracingEvaluator<Interval>>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn float_slice_tape(
        &self,
    ) -> <Self::FloatSliceEval as BulkEvaluator<f32>>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn grad_slice_tape(
        &self,
    ) -> <Self::GradSliceEval as BulkEvaluator<Grad>>::Tape;

    /// Computes a simplified tape using the given trace
    fn simplify(&self, trace: &Self::Trace) -> Result<Self, Error>
    where
        Self: Sized;

    /// Returns a size associated with this shape
    ///
    /// This is underspecified and only used for unit testing; for tape-based
    /// shapes, it's typically the length of the tape,
    #[cfg(test)]
    fn size(&self) -> usize;

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

/// A [`Shape`] which contains named variables
pub trait ShapeVars {
    /// Returns the variable map for ease of binding
    fn vars(&self) -> Arc<HashMap<String, u32>>;
}
