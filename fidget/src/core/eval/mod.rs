//! Traits and data structures for evaluation
//!
//! There are a bunch of things in here, but the most important trait is
//! [`Shape`], followed by the evaluator traits ([`BulkEvaluator`] and
//! [`TracingEvaluator`]).
//!
//! ```rust
//! use fidget::vm::VmShape;
//! use fidget::context::Context;
//! use fidget::eval::{TracingEvaluator, Shape, MathShape, EzShape};
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let shape = VmShape::new(&ctx, x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! let (value, _trace) = eval.eval(&tape, 0.25, 0.0, 0.0)?;
//! assert_eq!(value, 0.25);
//! # Ok::<(), fidget::Error>(())
//! ```
use crate::{
    context::Node,
    types::{Grad, Interval},
    Context, Error,
};

#[cfg(any(test, feature = "eval-tests"))]
pub mod test;

mod bulk;
mod tracing;
mod transform;

// Re-export a few things
pub use bulk::BulkEvaluator;
pub use tracing::TracingEvaluator;
pub use transform::TransformedShape;

/// A shape represents an implicit surface
///
/// It is mostly agnostic to _how_ that surface is represented; we simply
/// require that the shape can generate evaluators of various kinds.
///
/// Shapes are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
pub trait Shape: Send + Sync + Clone {
    /// Associated type traces collected during tracing evaluation
    ///
    /// This type must implement [`Eq`] so that traces can be compared; calling
    /// [`Shape::simplify`] with traces that compare equal should produce an
    /// identical result and may be cached.
    type Trace: Clone + Eq + Send + Trace;

    /// Associated type for storage used by the shape itself
    type Storage: Default + Send;

    /// Associated type for workspace used during shape simplification
    type Workspace: Default + Send;

    /// Associated type for storage used by tapes
    ///
    /// For simplicity, we require that every tape use the same type for storage.
    /// This could change in the future!
    type TapeStorage: Default + Send;

    /// Associated type for single-point tracing evaluation
    type PointEval: TracingEvaluator<
            Data = f32,
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
            Data = Interval,
            Trace = Self::Trace,
            TapeStorage = Self::TapeStorage,
        > + Send
        + Sync;

    /// Builds a new interval evaluator
    fn new_interval_eval() -> Self::IntervalEval {
        Self::IntervalEval::new()
    }

    /// Associated type for evaluating many points in one call
    type FloatSliceEval: BulkEvaluator<Data = f32, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new float slice evaluator
    fn new_float_slice_eval() -> Self::FloatSliceEval {
        Self::FloatSliceEval::new()
    }

    /// Associated type for evaluating many gradients in one call
    type GradSliceEval: BulkEvaluator<Data = Grad, TapeStorage = Self::TapeStorage>
        + Send
        + Sync;

    /// Builds a new gradient slice evaluator
    fn new_grad_slice_eval() -> Self::GradSliceEval {
        Self::GradSliceEval::new()
    }

    /// Returns an evaluation tape for a point evaluator
    fn point_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::PointEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn interval_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn float_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn grad_slice_tape(
        &self,
        storage: Self::TapeStorage,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape;

    /// Computes a simplified tape using the given trace, and reusing storage
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
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
    fn size(&self) -> usize;

    /// Associated type returned when applying a transform
    ///
    /// This is normally [`TransformedShape<Self>`](TransformedShape), but if
    /// `Self` is already `TransformedShape`, then the transform is stacked
    /// (instead of creating a wrapped object).
    type TransformedShape: Shape;

    /// Returns a shape with the given transform applied
    fn apply_transform(
        self,
        mat: nalgebra::Matrix4<f32>,
    ) -> <Self as Shape>::TransformedShape;
}

/// Extension trait for working with a shape without thinking much about memory
///
/// All of the [`Shape`] functions that use significant amounts of memory
/// pedantically require you to pass in storage for reuse.  This trait allows
/// you to ignore that, at the cost of performance; we require that all storage
/// types implement [`Default`], so these functions do the boilerplate for you.
///
/// This trait is automatically implemented for every [`Shape`], but must be
/// imported separately as a speed-bump to using it everywhere.
pub trait EzShape: Shape {
    /// Returns an evaluation tape for a point evaluator
    fn ez_point_tape(&self) -> <Self::PointEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for an interval evaluator
    fn ez_interval_tape(
        &self,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_float_slice_tape(
        &self,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape;

    /// Returns an evaluation tape for a float slice evaluator
    fn ez_grad_slice_tape(
        &self,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape;

    /// Computes a simplified tape using the given trace
    fn ez_simplify(&self, trace: &Self::Trace) -> Result<Self, Error>
    where
        Self: Sized;
}

impl<S: Shape> EzShape for S {
    fn ez_point_tape(&self) -> <Self::PointEval as TracingEvaluator>::Tape {
        self.point_tape(Default::default())
    }

    fn ez_interval_tape(
        &self,
    ) -> <Self::IntervalEval as TracingEvaluator>::Tape {
        self.interval_tape(Default::default())
    }

    fn ez_float_slice_tape(
        &self,
    ) -> <Self::FloatSliceEval as BulkEvaluator>::Tape {
        self.float_slice_tape(Default::default())
    }

    fn ez_grad_slice_tape(
        &self,
    ) -> <Self::GradSliceEval as BulkEvaluator>::Tape {
        self.grad_slice_tape(Default::default())
    }

    fn ez_simplify(&self, trace: &Self::Trace) -> Result<Self, Error> {
        let mut workspace = Default::default();
        self.simplify(trace, Default::default(), &mut workspace)
    }
}

/// A [`Shape`] which can be built from a math expression
pub trait MathShape {
    /// Builds a new shape from the given context and node
    fn new(ctx: &Context, node: Node) -> Result<Self, Error>
    where
        Self: Sized;

    /// Helper function to build a shape from a [`Tree`](crate::context::Tree)
    fn from_tree(t: &crate::context::Tree) -> Self
    where
        Self: Sized,
    {
        let mut ctx = Context::new();
        let node = ctx.import(t);
        Self::new(&ctx, node).unwrap()
    }
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

    /// Retrieves the internal storage from this tape
    fn recycle(self) -> Self::Storage;
}

/// Represents the trace captured by a tracing evaluation
///
/// The only property enforced on the trait is that we must have a way of
/// reusing trace allocations.  Because [`Trace`] implies `Clone` where it's
/// used in [`Shape`], this is trivial, but we can't provide a default
/// implementation because it would fall afoul of `impl` specialization.
pub trait Trace {
    /// Copies the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);
}

impl<T: Copy + Clone + Default> Trace for Vec<T> {
    fn copy_from(&mut self, other: &Self) {
        self.resize(other.len(), T::default());
        self.copy_from_slice(other);
    }
}
