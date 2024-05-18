//! Evaluates many points in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{eval::Tape, Error};

/// Trait for bulk evaluation returning the given type `T`
///
/// It's uncommon to use this trait outside the library itself; it's an
/// abstraction to reduce code duplication, and is public because it's used as a
/// constraint on other public APIs.
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
    /// Returns an error if the `x`, `y`, `z`, and `out` slices are of different
    /// lengths.
    fn eval(
        &mut self,
        tape: &Self::Tape,
        x: &[Self::Data],
        y: &[Self::Data],
        z: &[Self::Data],
    ) -> Result<&[Self::Data], Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }

    /// Helper function to return an error if the inputs are invalid
    fn check_arguments<T>(
        &self,
        xs: &[T],
        ys: &[T],
        zs: &[T],
        var_count: usize,
    ) -> Result<(), Error> {
        if xs.len() != ys.len() || ys.len() != zs.len() {
            Err(Error::MismatchedSlices)
        } else if var_count > 3 {
            Err(Error::BadVarSlice(3, var_count))
        } else {
            Ok(())
        }
    }
}
