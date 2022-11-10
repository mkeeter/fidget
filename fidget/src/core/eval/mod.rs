//! Traits and generic `struct`s for evaluation

mod choice;

pub mod float_slice;
pub mod grad;
pub mod interval;
pub mod point;

// Re-export a few things
pub use choice::Choice;

use float_slice::FloatSliceEvalT;
use grad::GradEvalT;
use interval::IntervalEvalT;
use point::PointEvalT;

/// Represents a "family" of evaluators (JIT, interpreter, etc)
pub trait Eval: Clone {
    /// Register limit for this evaluator family.
    const REG_LIMIT: u8;

    type IntervalEval: IntervalEvalT;
    type FloatSliceEval: FloatSliceEvalT;
    type PointEval: PointEvalT;
    type GradEval: GradEvalT;

    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];

    /// Builds a new point evaluator from the given `Tape`
    fn new_point_evaluator(tape: crate::tape::Tape) -> point::PointEval<Self> {
        point::PointEval::new(tape)
    }

    /// Builds a new interval evaluator from the given `Tape`
    fn new_interval_evaluator(
        tape: crate::tape::Tape,
    ) -> interval::IntervalEval<Self> {
        interval::IntervalEval::new(tape)
    }

    /// Builds a new interval evaluator from the given `Tape`, reusing storage
    fn new_interval_evaluator_with_storage(
        tape: crate::tape::Tape,
        storage: <<Self as Eval>::IntervalEval as IntervalEvalT>::Storage,
    ) -> interval::IntervalEval<Self> {
        interval::IntervalEval::new_with_storage(tape, storage)
    }

    /// Builds a float evaluator from the given `Tape`
    fn new_float_slice_evaluator(
        tape: crate::tape::Tape,
    ) -> float_slice::FloatSliceEval<Self> {
        float_slice::FloatSliceEval::new(tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_float_slice_evaluator_with_storage(
        tape: crate::tape::Tape,
        storage: <<Self as Eval>::FloatSliceEval as FloatSliceEvalT>::Storage,
    ) -> float_slice::FloatSliceEval<Self> {
        float_slice::FloatSliceEval::new_with_storage(tape, storage)
    }

    /// Builds a grad slice evaluator from the given `Tape`
    fn new_grad_evaluator(tape: crate::tape::Tape) -> grad::GradEval<Self> {
        grad::GradEval::new(tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_grad_evaluator_with_storage(
        tape: crate::tape::Tape,
        storage: <<Self as Eval>::GradEval as GradEvalT>::Storage,
    ) -> grad::GradEval<Self> {
        grad::GradEval::new_with_storage(tape, storage)
    }
}
