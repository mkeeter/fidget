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

    /// Associated type for storage used by the shape itself
    type Storage: Send;

    /// Associated type for storage used by tapes
    ///
    /// For simplicity, we require that every tape use the same type for storage.
    /// This could change in the future!
    type TapeStorage: Send;

    /// Associated type for single-point tracing evaluation
    type PointEval: TracingEvaluator<
            f32,
            Trace = Self::Trace,
            TapeStorage = Self::TapeStorage,
        > + Send
        + Sync;

    /// Builds a new point evaluator
    fn new_point_eval() -> Self::PointEval {
        Self::PointEval::new()
    }

    /// Associated type for single interval tracing evaluation
    type IntervalEval: TracingEvaluator<
            Interval,
            Trace = Self::Trace,
            TapeStorage = Self::TapeStorage,
        > + Send
        + Sync;

    /// Builds a new interval evaluator
    fn new_interval_eval() -> Self::IntervalEval {
        Self::IntervalEval::new()
    }

    /// Associated type for evaluating many points in one call
    type FloatSliceEval: BulkEvaluator<f32, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new float slice evaluator
    fn new_float_slice_eval() -> Self::FloatSliceEval {
        Self::FloatSliceEval::new()
    }

    /// Associated type for evaluating many gradients in one call
    type GradSliceEval: BulkEvaluator<Grad, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new gradient slice evaluator
    fn new_grad_slice_eval() -> Self::GradSliceEval {
        Self::GradSliceEval::new()
    }

    /// Returns an evaluation tape for a point evaluator
    fn point_tape(
        &self,
        storage: Option<Self::TapeStorage>,
    ) -> <Self::PointEval as TracingEvaluator<f32>>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn interval_tape(
        &self,
        storage: Option<Self::TapeStorage>,
    ) -> <Self::IntervalEval as TracingEvaluator<Interval>>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn float_slice_tape(
        &self,
        storage: Option<Self::TapeStorage>,
    ) -> <Self::FloatSliceEval as BulkEvaluator<f32>>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn grad_slice_tape(
        &self,
        storage: Option<Self::TapeStorage>,
    ) -> <Self::GradSliceEval as BulkEvaluator<Grad>>::Tape;

    /// Computes a simplified tape using the given trace, and reusing storage
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Option<Self::Storage>,
    ) -> Result<Self, Error>
    where
        Self: Sized;

    /// Attempt to reclaim storage from this shape
    ///
    /// This may fail, because shapes are `Clone` and are often implemented
    /// using an `Arc` around a heavier data structure.
    fn recycle(self) -> Option<Self::Storage>;

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

/// A tape represents something that can be evaluated by an evaluator
///
/// The only property enforced on the trait is that we must have some way to
/// recycle its internal storage.  This matters most for JIT evaluators, whose
/// tapes are regions of executable memory-mapped RAM (which is expensive to map
/// and unmap).
pub trait Tape {
    /// Associated type for this tape's data storage
    type Storage: Default;

    /// Attempt to retrieve the internal storage from this tape
    fn recycle(self) -> Self::Storage;
}
