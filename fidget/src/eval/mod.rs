//! Evaluation, both generically and with a small local interpreter

mod choice;
mod eval;
mod interval;
mod math;
mod traits;

pub use eval::{AsmFloatSliceEval, AsmFunc, AsmIntervalEval};
pub use interval::Interval;
pub use math::EvalMath;
pub use traits::{
    FloatEval, FloatFunc, FloatSliceEval, FloatSliceFunc, FromTape,
    IntervalEval, IntervalFunc,
};

pub(crate) use choice::Choice;
