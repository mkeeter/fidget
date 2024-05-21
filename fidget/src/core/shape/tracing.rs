//! Capturing a trace of function evaluation for further optimization
//!
//! Tracing evaluators are run on a single data type and capture a trace of
//! execution, which is the [`Trace` associated type](TracingEvaluator::Trace).
//!
//! The resulting trace can be used to simplify the original shape.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{eval::Tape, Error};

/// Evaluator for single values which simultaneously captures an execution trace
///
/// The trace can later be used to simplify the [`Shape`](crate::shape::Shape)
/// using [`Shape::simplify`](crate::shape::Shape::simplify).
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
    fn eval<F: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
    ) -> Result<(Self::Data, Option<&Self::Trace>), Error>;

    #[cfg(test)]
    fn eval_x<J: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: J,
    ) -> Self::Data {
        self.eval(tape, x.into(), Self::Data::from(0.0), Self::Data::from(0.0))
            .unwrap()
            .0
    }
    #[cfg(test)]
    fn eval_xy<J: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: J,
        y: J,
    ) -> Self::Data {
        self.eval(tape, x.into(), y.into(), Self::Data::from(0.0))
            .unwrap()
            .0
    }
}
