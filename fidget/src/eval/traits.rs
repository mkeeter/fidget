use crate::{eval::Interval, tape::Tape};

pub trait IntervalFuncHandle: From<Tape> {
    type Evaluator<'a>: IntervalEval<'a>
    where
        Self: 'a;
    fn get_evaluator(&self) -> Self::Evaluator<'_>;
}

pub trait IntervalEval<'a> {
    fn simplify(&self) -> Tape;
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval;
}

pub trait VecFuncHandle: From<Tape> {
    type Evaluator<'a>: VecEval<'a>
    where
        Self: 'a;
    fn get_evaluator(&self) -> Self::Evaluator<'_>;
}
pub trait VecEval<'a> {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4];
}

pub trait FloatFuncHandle: From<Tape> {
    type Evaluator<'a>: FloatEval<'a>
    where
        Self: 'a;
    fn get_evaluator(&self) -> Self::Evaluator<'_>;
}
pub trait FloatEval<'a> {
    fn eval(&mut self, x: f32, y: f32, z: f32) -> f32;
}

impl<'a, F: FloatEval<'a>> VecEval<'a> for F {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = FloatEval::eval(self, x[i], y[i], z[i])
        }
        out
    }
}
