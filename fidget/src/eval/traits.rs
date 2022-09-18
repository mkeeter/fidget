use crate::{eval::Interval, tape::Tape};

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `IntervalEval` objects, which actually do evaluation.
pub trait IntervalFuncHandle<'a>: Sync {
    type Recurse<'b>: IntervalFuncHandle<'b>;
    type Evaluator<'b>: IntervalEval<'b>
    where
        Self: 'b;

    fn get_evaluator(&self) -> Self::Evaluator<'_>;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// Interval evaluator
///
/// The evaluator will likely have a lifetime bounded to its parent
/// [`IntervalFuncHandle`](crate::eval::IntervalFuncHandle), and can generate
/// a new [`Tape`](crate::eval::Tape) on demand after evaluation.
pub trait IntervalEval<'a> {
    fn simplify(&self) -> Tape;
    fn eval_i(&mut self, x: Interval, y: Interval, z: Interval) -> Interval;
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for `f32 x 4` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `VecEval` objects, which actually do evaluation.
pub trait VecFuncHandle<'a> {
    type Recurse<'b>: VecFuncHandle<'b>;
    type Evaluator<'b>: VecEval<'b>
    where
        Self: 'b;

    fn get_evaluator(&self) -> Self::Evaluator<'_>;
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
/// one or more `VecEval` objects, which actually do evaluation.
pub trait FloatFuncHandle<'a> {
    type Recurse<'b>: FloatFuncHandle<'b>;
    type Evaluator<'b>: FloatEval<'b>
    where
        Self: 'b;

    fn get_evaluator(&self) -> Self::Evaluator<'_>;
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
