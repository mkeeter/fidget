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
        vars: &[&[Self::Data]],
    ) -> Result<&[Self::Data], Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }

    /// Helper function to return an error if the inputs are invalid
    fn check_arguments(
        &self,
        vars: &[&[Self::Data]],
        var_count: usize,
    ) -> Result<(), Error> {
        if var_count != vars.len() {
            Err(Error::BadVarSlice(vars.len(), var_count))
        } else {
            let Some(n) = vars.first().map(|v| v.len()) else {
                return Ok(());
            };
            if vars.iter().any(|v| v.len() == n) {
                Ok(())
            } else {
                Err(Error::MismatchedSlices)
            }
        }
    }
}
