//! Traits and data structures for evaluation
//!
//! The easiest way to build an evaluator of a particular kind is the
//! [`Eval`](Eval) extension trait on [`Family`](Family):
//!
//! ```rust
//! use fidget::eval::Eval;
//! use fidget::vm;
//! use fidget::context::Context;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let tape = ctx.get_tape(x).unwrap();
//!
//! // `vm::Eval` implements `Family`, so we can use it to build any kind of
//! // evaluator.  In this case, we'll build a single-point evaluator:
//! let mut eval = vm::Eval::new_point_evaluator(tape);
//! assert_eq!(eval.eval(0.25, 0.0, 0.0, &[]).unwrap().0, 0.25);
//! ```

// Bulk evaluators
pub mod float_slice;
pub mod grad_slice;

// Tracing evaluators
pub mod interval;
pub mod point;

pub mod bulk;
pub mod tape;
pub mod tracing;

mod vars;

// Re-export a few things
pub use float_slice::FloatSliceEval;
pub use grad_slice::Grad;
pub use grad_slice::GradSliceEval;
pub use interval::Interval;
pub use point::PointEval;
pub use tape::Tape;
pub use tracing::Choice;
pub use vars::Vars;

use bulk::BulkEvaluator;
use tracing::TracingEvaluator;

/// Represents a "family" of evaluators (JIT, interpreter, etc)
pub trait Family: Clone {
    /// Register limit for this evaluator family.
    const REG_LIMIT: u8;

    type PointEval: TracingEvaluator<f32, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send;
    type IntervalEval: TracingEvaluator<Interval, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send;

    type FloatSliceEval: BulkEvaluator<f32, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send;
    type GradSliceEval: BulkEvaluator<Grad, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send;

    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];
}

/// Helper trait used to add evaluator constructions to anything implementing
/// [`Family`](Family).
pub trait Eval<F: Family> {
    fn new_point_evaluator(tape: Tape<F>) -> point::PointEval<F>;

    fn new_interval_evaluator(tape: Tape<F>) -> interval::IntervalEval<F>;
    fn new_interval_evaluator_with_storage(
        tape: Tape<F>,
        storage: interval::IntervalEvalStorage<F>,
    ) -> interval::IntervalEval<F>;

    fn new_float_slice_evaluator(
        tape: Tape<F>,
    ) -> float_slice::FloatSliceEval<F>;
    fn new_float_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: float_slice::FloatSliceEvalStorage<F>,
    ) -> float_slice::FloatSliceEval<F>;

    fn new_grad_slice_evaluator(tape: Tape<F>) -> grad_slice::GradSliceEval<F>;
    fn new_grad_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: grad_slice::GradSliceEvalStorage<F>,
    ) -> grad_slice::GradSliceEval<F>;
}

impl<F: Family> Eval<F> for F {
    /// Builds a point evaluator from the given `Tape`
    fn new_point_evaluator(tape: Tape<F>) -> point::PointEval<F> {
        point::PointEval::new(&tape)
    }

    /// Builds an interval evaluator from the given `Tape`
    fn new_interval_evaluator(tape: Tape<F>) -> interval::IntervalEval<F> {
        interval::IntervalEval::new(&tape)
    }

    /// Builds an interval evaluator from the given `Tape`, reusing storage
    fn new_interval_evaluator_with_storage(
        tape: Tape<F>,
        storage: interval::IntervalEvalStorage<F>,
    ) -> interval::IntervalEval<F> {
        interval::IntervalEval::new_with_storage(&tape, storage)
    }

    /// Builds a float evaluator from the given `Tape`
    fn new_float_slice_evaluator(
        tape: Tape<F>,
    ) -> float_slice::FloatSliceEval<F> {
        float_slice::FloatSliceEval::new(&tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_float_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: float_slice::FloatSliceEvalStorage<F>,
    ) -> float_slice::FloatSliceEval<F> {
        float_slice::FloatSliceEval::new_with_storage(&tape, storage)
    }

    /// Builds a grad slice evaluator from the given `Tape`
    fn new_grad_slice_evaluator(tape: Tape<F>) -> grad_slice::GradSliceEval<F> {
        grad_slice::GradSliceEval::new(&tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_grad_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: grad_slice::GradSliceEvalStorage<F>,
    ) -> grad_slice::GradSliceEval<F> {
        grad_slice::GradSliceEval::new_with_storage(&tape, storage)
    }
}

/// Represents an evaluator with some internal (immutable) storage
///
/// For example, the JIT evaluators declare their allocated `mmap` data as their
/// `Storage`, which allows us to reuse pages.
pub trait EvaluatorStorage<F> {
    type Storage: Default;

    /// Constructs the evaluator, giving it a chance to reuse storage
    ///
    /// The incoming `Storage` is consumed, though it may not necessarily be
    /// used to construct the new tape (e.g. if it's a memory-mapped region and
    /// is too small).
    fn new_with_storage(tape: &Tape<F>, storage: Self::Storage) -> Self;

    /// Extract the internal storage for reuse, if possible
    fn take(self) -> Option<Self::Storage>;
}
