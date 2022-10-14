use crate::tape::Tape;

/// Function handle for evaluation of many points simultaneously.
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub trait FloatSliceFuncT {
    type Evaluator: FloatSliceEvalT;

    /// Storage used by the type
    type Storage;

    /// Builds the handle.
    fn from_tape(tape: Tape) -> Self;

    /// Constructs the `FloatSliceFuncT`, giving it a chance to reuse storage
    ///
    /// If the `storage` argument is used, then it's consumed; otherwise, it's
    /// returned as part of the tuple.
    fn from_tape_give(
        tape: Tape,
        storage: Self::Storage,
    ) -> (Self, Option<Self::Storage>)
    where
        Self: Sized;

    /// Returns an evaluator, which may borrow from this handle
    ///
    /// This should be an O(1) operation; heavy lifting should have been
    /// previously done when constructing the `FloatSliceFuncT` itself.
    fn get_evaluator(&self) -> Self::Evaluator;

    /// Extract the internal storage for reuse
    fn take(self) -> Option<Self::Storage>;
}

/// Simultaneous evaluation of many points
pub trait FloatSliceEvalT {
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub struct FloatSliceFunc<F> {
    tape: Tape,
    func: F,
}

impl<F: FloatSliceFuncT> FloatSliceFunc<F> {
    pub fn from_tape(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            func: F::from_tape(tape),
        }
    }
    pub fn new_give(tape: Tape, s: F::Storage) -> (Self, Option<F::Storage>) {
        let (func, out) = F::from_tape_give(tape.clone(), s);
        (Self { tape, func }, out)
    }

    pub fn get_evaluator(
        &self,
    ) -> FloatSliceEval<<F as FloatSliceFuncT>::Evaluator> {
        FloatSliceEval {
            tape: self.tape.clone(),
            eval: self.func.get_evaluator(),
        }
    }
    pub fn take(self) -> Option<F::Storage> {
        self.func.take()
    }
}

pub struct FloatSliceEval<E> {
    #[allow(dead_code)]
    pub(crate) tape: Tape,
    pub(crate) eval: E,
}

impl<E: FloatSliceEvalT> FloatSliceEval<E> {
    pub fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
        self.eval.eval_s(x, y, z, out)
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
