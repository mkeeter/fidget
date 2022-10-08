//! Evaluation, both generically and with a small local interpreter

mod asm;
mod choice;

pub mod float_slice;
pub mod interval;
pub mod point;

// Re-export a few things
pub use asm::AsmFamily;
pub(crate) use choice::Choice;

use crate::tape::Tape;

/// Represents a "family" of evaluators (JIT, interpreter, etc)
///
/// This trait is only needed as a work-around to allow recursion during
/// rendering.  It should be implemented on an unutterable type (e.g. `enum
/// MyEvalFamily {}`).
pub trait EvalFamily<'a> {
    /// Register limit for this evaluator family.
    const REG_LIMIT: u8;

    type IntervalFunc: interval::IntervalFuncT<'a>;
    type FloatSliceFunc: float_slice::FloatSliceFuncT<'a>;
    type PointFunc: point::PointFuncT<'a>;

    fn from_tape_i_inner(t: &'a Tape) -> Self::IntervalFunc;
    fn from_tape_s_inner(t: &'a Tape) -> Self::FloatSliceFunc;
    fn from_tape_p_inner(t: &'a Tape) -> Self::PointFunc;

    fn from_tape_i(
        tape: &'a Tape,
    ) -> interval::IntervalFunc<'a, Self::IntervalFunc> {
        interval::IntervalFunc::new(tape, Self::from_tape_i_inner(tape))
    }
    fn from_tape_s(
        tape: &'a Tape,
    ) -> float_slice::FloatSliceFunc<'a, Self::FloatSliceFunc> {
        float_slice::FloatSliceFunc::new(tape, Self::from_tape_s_inner(tape))
    }
    fn from_tape_p(tape: &'a Tape) -> point::PointFunc<'a, Self::PointFunc> {
        point::PointFunc::new(tape, Self::from_tape_p_inner(tape))
    }
}
