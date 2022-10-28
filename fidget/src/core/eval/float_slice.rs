use crate::{eval::EvalFamily, tape::Tape};

/// Function handle for evaluation of many points simultaneously.
pub trait FloatSliceEvalT: From<Tape> {
    /// Storage used by the type
    type Storage;

    /// Parent family (used to get `REG_LIMIT`)
    type Family: EvalFamily;

    /// Constructs the `FloatSliceT`, giving it a chance to reuse storage
    ///
    /// The incoming `Storage` is consumed, though it may not necessarily be
    /// used to construct the new tape (e.g. if it's a mmap region and is too
    /// small).
    fn from_tape_give(tape: Tape, storage: Self::Storage) -> Self
    where
        Self: Sized;

    /// Extract the internal storage for reuse, if possible
    fn take(self) -> Option<Self::Storage>;

    /// Evaluates float slices
    // TODO: make this return an error if sizes are mismatched?
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

pub struct FloatSliceEval<E> {
    #[allow(dead_code)]
    tape: Tape,
    eval: E,
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
    pub fn new_give(tape: Tape, s: F::Storage) -> Self {
        let eval = F::from_tape_give(tape.clone(), s);
        Self { tape, eval }
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

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::context::Context;

    pub fn test_give_take<I: FloatSliceEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape_x = ctx.get_tape(x, I::Family::REG_LIMIT);
        let tape_y = ctx.get_tape(y, I::Family::REG_LIMIT);

        let eval = FloatSliceEval::<I>::from(tape_y.clone());
        let mut out = [0.0; 4];
        let mut t = eval.take().unwrap();

        // This is a fuzz test for icache issues
        for _ in 0..10000 {
            let (mut eval, s) =
                FloatSliceEval::<I>::new_give(tape_x.clone(), t);
            assert!(s.is_none());
            eval.eval_s(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &mut out,
            );
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
            t = eval.take().unwrap();

            let (mut eval, s) =
                FloatSliceEval::<I>::new_give(tape_y.clone(), t);
            assert!(s.is_none());
            eval.eval_s(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &mut out,
            );
            assert_eq!(out, [3.0, 2.0, 1.0, 0.0]);
            t = eval.take().unwrap();
        }
    }

    pub fn test_vectorized<I: FloatSliceEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape(x, I::Family::REG_LIMIT);
        let mut eval = FloatSliceEval::<I>::from(tape);
        let mut out = [0.0; 4];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &mut out,
        );
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let two = ctx.constant(2.0);
        let mul = ctx.mul(y, two).unwrap();
        let tape = ctx.get_tape(mul, I::Family::REG_LIMIT);
        let mut eval = FloatSliceEval::<I>::from(tape);
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &mut out,
        );
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        eval.eval_s(
            &[0.0, 1.0, 2.0],
            &[1.0, 4.0, 8.0],
            &[0.0, 0.0, 0.0],
            &mut out[0..3],
        );
        assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

        // out is longer than inputs
        eval.eval_s(
            &[0.0, 1.0, 2.0],
            &[1.0, 4.0, 4.0],
            &[0.0, 0.0, 0.0],
            &mut out[0..4],
        );
        assert_eq!(&out[0..3], &[2.0, 8.0, 8.0]);

        let mut out = [0.0; 7];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
            &[0.0; 7],
            &mut out,
        );
        assert_eq!(out, [2.0, 8.0, 8.0, -2.0, -4.0, -6.0, 0.0]);
    }

    #[macro_export]
    macro_rules! float_slice_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::float_slice::eval_tests::$i::<$t>()
            }
        };
    }

    #[macro_export]
    macro_rules! float_slice_tests {
        ($t:ty) => {
            $crate::float_slice_test!(test_give_take, $t);
            $crate::float_slice_test!(test_vectorized, $t);
        };
    }
}
