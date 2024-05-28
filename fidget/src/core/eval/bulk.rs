//! Evaluates many points in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{eval::Tape, Error};

/// Trait for bulk evaluation returning the given type `T`
pub trait BulkEvaluator: Default {
    /// Data type used during evaluation
    type Data: From<f32> + Copy + Clone;

    /// Instruction tape used during evaluation
    ///
    /// This may be a literal instruction tape (in the case of VM evaluation),
    /// or a metaphorical instruction tape (e.g. a JIT function).
    type Tape: Tape<Storage = Self::TapeStorage> + Send + Sync;

    /// Associated type for tape storage
    ///
    /// This is a workaround for plumbing purposes
    type TapeStorage;

    /// Evaluates many points using the given instruction tape
    ///
    /// `vars` should be a slice-of-slices (or a slice-of-`Vec`s) representing
    /// input arguments for each of the tape's variables; use [`Tape::vars`] to
    /// map from [`Var`](crate::var::Var) to position in the list.
    ///
    /// Returns an error if any of the `var` slices are of different lengths, or
    /// if all variables aren't present.
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<&[Self::Data], Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }
}
