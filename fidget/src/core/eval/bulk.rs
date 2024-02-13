//! Evaluates many points in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! A bulk evaluator expects to be given **many single points**, i.e. the X, Y,
//! Z inputs are always `&[f32]`.  The output may be of a different type, e.g.
//! partial derivatives with respect to X/Y/Z
//! ([`GradEval`](crate::eval::GradSliceEval)).
//!
//! Bulk evaluators are typically named `XSliceEval`, where `X` is the output
//! type.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::Error;

/// Trait for bulk evaluation returning the given type `T`
///
/// It's uncommon to use this trait outside the library itself; it's an
/// abstraction to reduce code duplication, and is public because it's used as a
/// constraint on other public APIs.
pub trait BulkEvaluator<T>: Default {
    /// Instruction tape used during evaluation
    ///
    /// This may be a literal instruction tape (in the case of VM evaluation),
    /// or a metaphorical instruction tape (e.g. a JIT function).
    type Tape: Send + Sync;

    /// Evaluates many points using the given instruction tape
    ///
    /// # Panics
    /// This function may assume that the `x`, `y`, `z`, and `out` slices are of
    /// equal length and panic otherwise; higher-level calls should maintain
    /// that invariant.
    ///
    /// This function may also assume that `vars` is correctly sized for the
    /// number of variables in the tape.
    fn eval(
        &mut self,
        tape: &Self::Tape,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
    ) -> Result<&[T], Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }
}
