use crate::{
    eval::{Choice, Interval},
    tape::Tape,
};

pub trait IntervalFuncHandle {
    type Evaluator: IntervalEval;
    fn get_raw_evaluator(&self) -> Self::Evaluator;
    fn get_evaluator(&self) -> OwnedIntervalEval<Self::Evaluator> {
        OwnedIntervalEval {
            eval: self.get_raw_evaluator(),
            _p: std::marker::PhantomData,
        }
    }
}

pub struct OwnedIntervalEval<'a, T: IntervalEval> {
    eval: T,
    _p: std::marker::PhantomData<&'a ()>,
}

pub trait IntervalEval {
    fn choices(&self) -> &[Choice];
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval;
}

impl<'a, T: IntervalEval> IntervalEval for OwnedIntervalEval<'a, T> {
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval {
        self.eval.eval(x, y, z)
    }
    fn choices(&self) -> &[Choice] {
        self.eval.choices()
    }
}

/*
pub trait FloatFuncHandle {
    type Evaluator: FloatEval;
    fn get_evaluator(&self) -> Self::Evaluator;
}

pub trait FloatEval {
    fn eval(&mut self, x: f32, y: f32, z: f32) -> f32;
}
*/

pub trait VecFuncHandle {
    type Evaluator: VecEval;
    fn get_raw_evaluator(&self) -> Self::Evaluator;
    fn get_evaluator(&self) -> OwnedVecEval<Self::Evaluator> {
        OwnedVecEval {
            eval: self.get_raw_evaluator(),
            _p: std::marker::PhantomData,
        }
    }
}

pub trait VecEval {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4];
}

pub struct OwnedVecEval<'a, T: VecEval> {
    eval: T,
    _p: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: VecEval> VecEval for OwnedVecEval<'a, T> {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        self.eval.eval(x, y, z)
    }
}
