//! Traits and data structures for function evaluation
use crate::{
    Error,
    context::{Context, Node},
    types::{Grad, Interval},
    var::VarMap,
};

#[cfg(any(test, feature = "eval-tests"))]
#[allow(missing_docs)]
pub mod test;

mod bulk;
mod tracing;

// Reexport a few types
pub use bulk::{BulkEvaluator, BulkOutput};
pub use tracing::TracingEvaluator;

/// A tape represents something that can be evaluated by an evaluator
///
/// It includes some kind of storage (which could be empty) and the ability to
/// look up variable mapping.
///
/// Tapes may be shared between threads, so they should be cheap to clone (i.e.
/// a wrapper around an `Arc<..>`).
pub trait Tape: Send + Sync + Clone {
    /// Associated type for this tape's data storage
    type Storage: Default;

    /// Tries to retrieve the internal storage from this tape
    ///
    /// This matters most for JIT evaluators, whose tapes are regions of
    /// executable memory-mapped RAM (which is expensive to map and unmap).
    fn recycle(self) -> Option<Self::Storage>;

    /// Returns a mapping from [`Var`](crate::var::Var) to evaluation index
    ///
    /// This must be identical to [`Function::vars`] on the `Function` which
    /// produced this tape.
    fn vars(&self) -> &VarMap;

    /// Returns the number of outputs written by this tape
    ///
    /// The order of outputs is set by the caller at tape construction, so we
    /// don't need a map to determine the index of a particular output (unlike
    /// variables).
    fn output_count(&self) -> usize;
}

/// Represents the trace captured by a tracing evaluation
///
/// The only property enforced on the trait is that we must have a way of
/// reusing trace allocations.  Because [`Trace`] implies `Clone` where it's
/// used in [`Function`], this is trivial, but we can't provide a default
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

/// A function represents something that can be evaluated
///
/// It is mostly agnostic to _how_ that something is represented; we simply
/// require that it can generate evaluators of various kinds.
///
/// Inputs to the function should be represented as [`Var`](crate::var::Var)
/// values; the [`vars()`](Function::vars) function returns the mapping from
/// `Var` to position in the input slice.
///
/// Functions are shared between threads, so they should be cheap to clone.  In
/// most cases, they're a thin wrapper around an `Arc<..>`.
pub trait Function: Send + Sync + Clone {
    /// Associated type traces collected during tracing evaluation
    ///
    /// This type must implement [`Eq`] so that traces can be compared; calling
    /// [`Function::simplify`] with traces that compare equal should produce an
    /// identical result and may be cached.
    type Trace: Clone + Eq + Send + Sync + Trace;

    /// Associated type for storage used by the function itself
    type Storage: Default + Send;

    /// Associated type for workspace used during function simplification
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

    /// Attempt to reclaim storage from this function
    ///
    /// This may fail, because functions are `Clone` and are often implemented
    /// using an `Arc` around a heavier data structure.
    fn recycle(self) -> Option<Self::Storage>;

    /// Returns a size associated with this function
    ///
    /// This is underspecified and only used for unit testing; for tape-based
    /// functions, it's typically the length of the tape,
    fn size(&self) -> usize;

    /// Returns the map from [`Var`](crate::var::Var) to input index
    fn vars(&self) -> &VarMap;

    /// Checks to see whether this function can ever be simplified
    fn can_simplify(&self) -> bool;
}

/// A [`Function`] which can be built from a math expression
pub trait MathFunction: Function {
    /// Builds a new function from the given context and node
    fn new(ctx: &Context, nodes: &[Node]) -> Result<Self, Error>
    where
        Self: Sized;
}
