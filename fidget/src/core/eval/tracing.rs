//! Capturing a trace of function evaluation for further optimization
//!
//! Tracing evaluators are run on a single data type and capture a trace of
//! execution, recording decision points in a [`Vec<Choice>`](Choice).  Decision
//! points are places where the code could take a single branch out of multiple
//! options; for example, a `min` or `max` node.
//!
//! The resulting trace can be used to simplify the instruction tape.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::Error;

/// A single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
///
/// Note that this is a bitfield such that
/// ```rust
/// # use fidget::eval::Choice;
/// # assert!(
/// Choice::Both as u8 == Choice::Left as u8 | Choice::Right as u8
/// # );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    /// This choice has not yet been assigned
    ///
    /// A value of `Unknown` is invalid after evaluation
    Unknown = 0,

    /// The operation always picks the left-hand input
    Left = 1,

    /// The operation always picks the right-hand input
    Right = 2,

    /// The operation may pick either input
    Both = 3,
}

impl std::ops::BitOrAssign<Choice> for Choice {
    fn bitor_assign(&mut self, other: Self) {
        *self = match (*self as u8) | (other as u8) {
            0 => Self::Unknown,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Both,
            _ => unreachable!(),
        }
    }
}

/// A tracing evaluator performs evaluation of a single `T`, capturing a trace
/// of execution for further simplification.
///
/// This trait is unlikely to be used directly; instead, use a [`TracingEval`],
/// which offers a user-friendly API.
pub trait TracingEvaluator<T: From<f32>, Trace>: Default {
    /// Instruction tape used during evaluation
    ///
    /// This may be a literal instruction tape (in the case of VM evaluation),
    /// or a metaphorical instruction tape (e.g. a JIT function).
    type Tape: Send + Sync;

    /// Evaluates the given value, using `choices` and `data` as scratch memory.
    fn eval<F: Into<T>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(T, Option<&Trace>), Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn eval_x<J: Into<T>>(&mut self, tape: &Self::Tape, x: J) -> T {
        self.eval(tape, x.into(), T::from(0.0), T::from(0.0), &[])
            .unwrap()
            .0
    }
    #[cfg(test)]
    fn eval_xy<J: Into<T>>(&mut self, tape: &Self::Tape, x: J, y: J) -> T {
        self.eval(tape, x.into(), y.into(), T::from(0.0), &[])
            .unwrap()
            .0
    }
}
