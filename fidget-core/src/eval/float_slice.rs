use crate::tape::Tape;

/// Function handle for evaluation of many points simultaneously.
pub trait FloatSliceEvalT: From<Tape> {
    /// Storage used by the type
    type Storage;

    /// Constructs the `FloatSliceT`, giving it a chance to reuse storage
    ///
    /// If the `storage` argument is used, then it's consumed; otherwise, it's
    /// returned as part of the tuple.
    fn from_tape_give(
        tape: Tape,
        storage: Self::Storage,
    ) -> (Self, Option<Self::Storage>)
    where
        Self: Sized;

    /// Extract the internal storage for reuse
    fn take(self) -> Option<Self::Storage>;

    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

pub struct FloatSliceEval<E> {
    #[allow(dead_code)]
    pub(crate) tape: Tape,
    pub(crate) eval: E,
}

impl<E: FloatSliceEvalT> From<Tape> for FloatSliceEval<E> {
    fn from(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            eval: E::from(tape),
        }
    }
}

impl<F: FloatSliceEvalT> FloatSliceEval<F> {
    pub fn new_give(tape: Tape, s: F::Storage) -> (Self, Option<F::Storage>) {
        let (eval, out) = F::from_tape_give(tape.clone(), s);
        (Self { tape, eval }, out)
    }

    pub fn take(self) -> Option<F::Storage> {
        self.eval.take()
    }

    pub fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
        self.eval.eval_s(x, y, z, out)
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
