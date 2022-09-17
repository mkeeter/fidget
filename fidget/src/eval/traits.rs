use crate::{eval::Interval, tape::Tape};

pub trait IntervalFuncHandle: From<Tape> {
    type Evaluator<'a>: IntervalEval<'a>
    where
        Self: 'a;
    fn get_evaluator(&self) -> Self::Evaluator<'_>;
}

pub trait IntervalEval<'a> {
    fn simplify(&self) -> Tape;
    fn eval(&mut self, x: [f32; 2], y: [f32; 2], z: [f32; 2]) -> [f32; 2];
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
