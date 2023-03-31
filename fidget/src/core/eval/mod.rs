//! Traits and data structures for evaluation
//!
//! The easiest way to build an evaluator of a particular kind is by calling
//! `new_X_evaluator` on [`Tape`](Tape).
//!
//! ```rust
//! use fidget::vm;
//! use fidget::context::Context;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let tape = ctx.get_tape::<vm::Eval>(x)?;
//!
//! // Let's build a single point evaluator:
//! let mut eval = tape.new_point_evaluator();
//! assert_eq!(eval.eval(0.25, 0.0, 0.0, &[])?.0, 0.25);
//! # Ok::<(), fidget::Error>(())
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
pub mod types;

mod vars;

// Re-export a few things
pub use float_slice::FloatSliceEval;
pub use grad_slice::GradSliceEval;
pub use interval::IntervalEval;
pub use point::PointEval;
pub use tape::Tape;
pub use tracing::Choice;
pub use vars::Vars;

use bulk::BulkEvaluator;
use tracing::TracingEvaluator;

/// A "family" of evaluators (JIT, interpreter, etc)
pub trait Family: Clone {
    /// Register limit for this evaluator family.
    const REG_LIMIT: u8;

    /// Single-point evaluator
    type PointEval: TracingEvaluator<f32, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send
        + Sync;
    /// Interval evaluator
    type IntervalEval: TracingEvaluator<types::Interval, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send
        + Sync;

    /// Bulk point evaluator
    type FloatSliceEval: BulkEvaluator<f32, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send
        + Sync;
    /// Bulk gradient evaluator
    type GradSliceEval: BulkEvaluator<types::Grad, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send
        + Sync;

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

/// An evaluator with some internal (immutable) storage
///
/// For example, the JIT evaluators declare their allocated `mmap` data as their
/// `Storage`, which allows us to reuse pages.
pub trait EvaluatorStorage<F> {
    /// Storage type associated with this evaluator
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
