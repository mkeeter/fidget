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

use crate::{eval::Tape, Error};

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
    fn eval<F: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(Self::Data, Option<&Self::Trace>), Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }

    /// Helper function to check input arguments
    fn check_arguments(
        &self,
        vars: &[f32],
        var_count: usize,
    ) -> Result<(), Error> {
        if vars.len() != var_count {
            Err(Error::BadVarSlice(vars.len(), var_count))
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    fn eval_x<J: Into<Self::Data>>(
        &mut self,
        tape: &Self::Tape,
        x: J,
    ) -> Self::Data {
        self.eval(
            tape,
            x.into(),
            Self::Data::from(0.0),
            Self::Data::from(0.0),
            &[],
        )
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
        self.eval(tape, x.into(), y.into(), Self::Data::from(0.0), &[])
            .unwrap()
            .0
    }
}
