use crate::{
    eval::{Choice, Interval},
    tape::Tape,
};

pub struct FuncHandle<'t, H> {
    tape: &'t Tape,
    handle: H,
}

impl<'t, H: for<'b> From<&'b Tape>> FuncHandle<'t, H> {
    pub fn new(t: &'t Tape) -> Self {
        Self {
            tape: t,
            handle: H::from(t),
        }
    }
}

impl<'t, H: IntervalFuncHandle + for<'b> From<&'b Tape>> FuncHandle<'t, H> {
    pub fn get_evaluator(&self) -> OwnedIntervalEval<'t, '_, H::Evaluator> {
        OwnedIntervalEval {
            tape: self.tape,
            eval: self.get_raw_evaluator(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<'t, H: IntervalFuncHandle> IntervalFuncHandle for FuncHandle<'t, H> {
    type Evaluator = H::Evaluator;
    fn get_raw_evaluator(&self) -> Self::Evaluator {
        self.handle.get_raw_evaluator()
    }
}

pub trait IntervalFuncHandle {
    type Evaluator: IntervalEval;
    fn get_raw_evaluator(&self) -> Self::Evaluator;
}

pub struct OwnedIntervalEval<'t, 'a, T: IntervalEval> {
    pub tape: &'t Tape,
    eval: T,
    _p: std::marker::PhantomData<&'a ()>,
}

pub trait IntervalEval {
    fn choices(&self) -> &[Choice];
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval;
}

impl<'t, 'a, T: IntervalEval> IntervalEval for OwnedIntervalEval<'t, 'a, T> {
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval {
        self.eval.eval(x, y, z)
    }
    fn choices(&self) -> &[Choice] {
        self.eval.choices()
    }
}

impl<'t, 'a, T: IntervalEval> OwnedIntervalEval<'t, 'a, T> {
    pub fn simplify(&self) -> Tape {
        self.tape.simplify(self.choices())
    }
}

pub trait FloatFuncHandle {
    type Evaluator: FloatEval;
    fn get_raw_evaluator(&self) -> Self::Evaluator;
    fn get_evaluator(&self) -> OwnedFloatEval<Self::Evaluator> {
        OwnedFloatEval {
            eval: self.get_raw_evaluator(),
            _p: std::marker::PhantomData,
        }
    }
}

pub trait FloatEval {
    fn eval(&mut self, x: f32, y: f32, z: f32) -> f32;
}

pub struct OwnedFloatEval<'a, T: FloatEval> {
    eval: T,
    _p: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: FloatEval> FloatEval for OwnedFloatEval<'a, T> {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        self.eval.eval(x, y, z)
    }
}

pub trait VecFuncHandle: for<'a> From<&'a Tape> {
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

impl<T: FloatEval> VecEval for T {
    fn eval(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = self.eval.eval(x[i], y[i], z[i]);
        }
        out
    }
}
