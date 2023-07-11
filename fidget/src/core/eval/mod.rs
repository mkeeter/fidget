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
pub mod tracing;
pub mod types;

mod vars;

// Re-export a few things
pub use float_slice::FloatSliceEval;
pub use grad_slice::GradSliceEval;
pub use interval::IntervalEval;
pub use point::PointEval;
pub use vars::Vars;

use bulk::BulkEvaluator;
use tracing::TracingEvaluator;

use crate::vm::Tape;

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

    /// Associated tape data
    ///
    /// For the VM evaluator, this is a `Vec<Op>`; for the JIT evaluator, this
    /// is a raw block of memory filled with executable code.
    type TapeData;

    /// Associated type for group metadata
    ///
    /// For the VM evaluator, this is a slice range into the [`TapeData`] (which
    /// is a `Vec<Op>`); for the JIT evaluator, it's a raw pointer.
    type GroupMetadata;

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
/// For example, the JIT evaluators declare their `Vec<*const u8>` as their
/// `Storage`, which allows us to reuse allocations.
pub trait EvaluatorStorage<F> {
    /// Storage type associated with this evaluator
    type Storage: Default + Send;

    /// Constructs the evaluator, giving it a chance to reuse storage
    fn new_with_storage(tape: &Tape<F>, storage: Self::Storage) -> Self;

    /// Extract the internal storage for reuse, if possible
    fn take(self) -> Option<Self::Storage>;
}
