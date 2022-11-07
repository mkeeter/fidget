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

    type IntervalEval: IntervalEvalT + Clone;
    type FloatSliceEval: FloatSliceEvalT;
    type PointEval: PointEvalT;
    type GradEval: GradEvalT;

    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];
}
