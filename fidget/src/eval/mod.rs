//! Evaluation, both generically and with a small local interpreter

mod choice;
mod eval;
mod float4;
mod interval;
mod math;
mod traits;

pub use eval::{AsmEval, InterpreterHandle};
pub use float4::Float4;
pub use interval::Interval;
pub use math::EvalMath;
pub use traits::{
    FloatEval, FloatFunc, IntervalEval, IntervalFunc, VecEval, VecFunc,
};

pub(crate) use choice::Choice;
