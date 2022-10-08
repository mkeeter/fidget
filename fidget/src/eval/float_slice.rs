use crate::tape::Tape;

/// Function handle for evaluation of many points simultaneously.
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub trait FloatSliceFuncT<'a> {
    type Evaluator: FloatSliceEvalT<'a>;

    /// Returns an evaluator, which may borrow from this handle
    ///
    /// This should be an O(1) operation; heavy lifting should have been
    /// previously done when constructing the `FloatSliceFuncT` itself.
    fn get_evaluator(&self) -> Self::Evaluator;
}

/// Simultaneous evaluation of many points
pub trait FloatSliceEvalT<'a> {
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub struct FloatSliceFunc<'a, F> {
    tape: &'a Tape,
    func: F,
}

impl<'a, F: FloatSliceFuncT<'a>> FloatSliceFunc<'a, F> {
    pub fn new(tape: &'a Tape, func: F) -> Self {
        Self { tape, func }
    }
    pub fn get_evaluator(&self) -> FloatSliceEval<'a, F::Evaluator> {
        FloatSliceEval {
            tape: self.tape,
            eval: self.func.get_evaluator(),
        }
    }
}

pub struct FloatSliceEval<'a, E> {
    #[allow(dead_code)]
    pub(crate) tape: &'a Tape,
    pub(crate) eval: E,
}

impl<'a, E: FloatSliceEvalT<'a>> FloatSliceEval<'a, E> {
    pub fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
        self.eval.eval_s(x, y, z, out)
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
