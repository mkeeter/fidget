//! Capturing a trace of function evaluation for further optimization
//!
//! Tracing evaluators are run on a single data type and capture a trace of
//! execution, which is the [`Trace` associated type](TracingEvaluator::Trace).
//!
//! The resulting trace can be used to simplify the original function.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{eval::Tape, Error};

/// Evaluator for single values which simultaneously captures an execution trace
///
/// The trace can later be used to simplify the [`Shape`](crate::eval::Shape)
/// using [`Shape::simplify`](crate::eval::Shape::simplify).
pub trait TracingEvaluator: Default {
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

    /// Associated type for the trace captured during evaluation
    type Trace;

    /// Evaluates the given tape at a particular position
    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[Self::Data],
    ) -> Result<(Self::Data, Option<&Self::Trace>), Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }

    /// Helper function to return an error if the inputs are invalid
    fn check_arguments(
        &self,
        vars: &[Self::Data],
        var_count: usize,
    ) -> Result<(), Error> {
        // It's fine if the caller has given us extra variables (e.g. due to
        // tape simplification), but it must have given us enough.
        if vars.len() < var_count {
            Err(Error::BadVarSlice(vars.len(), var_count))
        } else {
            Ok(())
        }
    }
}
