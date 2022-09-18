use crate::{eval::Interval, tape::Tape};

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `IntervalEval` objects, which actually do evaluation.
pub trait IntervalFunc<'a>: Sync {
    type Evaluator: IntervalEval<'a>;
    type Recurse<'b>: IntervalFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// Interval evaluator
///
/// The evaluator will likely have a lifetime bounded to its parent
/// [`IntervalFunc`](crate::eval::IntervalFunc), and can generate
/// a new [`Tape`](crate::tape::Tape) on demand after evaluation.
pub trait IntervalEval<'a> {
    fn simplify(&self) -> Tape;
    fn eval_i<I: Into<Interval>>(&mut self, x: I, y: I, z: I) -> Interval;
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for `f32 x 4` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `VecEval` objects, which actually do evaluation.
pub trait VecFunc<'a> {
    type Evaluator: VecEval<'a>;
    type Recurse<'b>: VecFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// `f32 x 4` evaluator
pub trait VecEval<'a> {
    fn eval_v(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4];
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for `f32` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatEval` objects, which actually do evaluation.
pub trait FloatFunc<'a> {
    type Evaluator: FloatEval<'a>;
    type Recurse<'b>: FloatFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// `f32` evaluator
pub trait FloatEval<'a> {
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32;
}

impl<'a, F: FloatEval<'a>> VecEval<'a> for F {
    fn eval_v(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = FloatEval::eval_f(self, x[i], y[i], z[i])
        }
        out
    }
}
