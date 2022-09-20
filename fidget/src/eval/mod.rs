//! Evaluation, both generically and with a small local interpreter

mod choice;
mod eval;
mod interval;
mod traits;

pub use eval::{AsmFloatSliceEval, AsmFunc, AsmIntervalEval};
pub use interval::Interval;
pub use traits::{
    EvalFamily, FloatEval, FloatFunc, FloatSliceEval, FloatSliceFunc,
    IntervalEval, IntervalFunc,
};

pub(crate) use choice::Choice;
