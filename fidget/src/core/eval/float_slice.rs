//! Float slice evaluation (i.e. `&[f32]`)
use crate::{
    eval::{tape::Tape, Eval},
    Error,
};

/// Function handle for evaluation of many points simultaneously.
pub trait FloatSliceEvalT<R> {
    /// Storage used by the evaluator, provided to minimize allocation churn
    type Storage: Default;

    fn new(tape: Tape<R>) -> Self;

    /// Constructs the `FloatSliceT`, giving it a chance to reuse storage
    ///
    /// In the default implementation, `_storage` is ignored; override this
    /// function if it would be useful.
    ///
    /// The incoming `Storage` is consumed, though it may not necessarily be
    /// used to construct the new tape (e.g. if it's a memory-mapped region and
    /// is too small).
    fn new_with_storage(tape: Tape<R>, _storage: Self::Storage) -> Self
    where
        Self: Sized,
    {
        Self::new(tape)
    }

    /// Extract the internal storage for reuse, if possible
    ///
    /// In the default implementation, this returns a default-constructed
    /// `Storage`; override this function if it would be useful
    fn take(self) -> Option<Self::Storage>
    where
        Self: Sized,
    {
        Some(Default::default())
    }

    /// Evaluates float slices
    ///
    /// # Panics
    /// This function may assume that the slices are of equal length and
    /// panic otherwise; higher-level calls (e.g.
    /// [`FloatSliceEval::eval_s`](`FloatSliceEval::eval_s`)) should maintain
    /// that invariant.
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

/// Evaluator for float slices, parameterized by evaluator family
pub struct FloatSliceEval<E: Eval> {
    #[allow(dead_code)]
    tape: Tape<E>,
    eval: E::FloatSliceEval,
}

impl<E: Eval> FloatSliceEval<E> {
    pub fn new(tape: Tape<E>) -> Self {
        Self {
            tape: tape.clone(),
            eval: E::FloatSliceEval::new(tape),
        }
    }

    /// Builds a new [`FloatSliceEval`](Self), reusing storage to minimize churn
    pub fn new_with_storage(
        tape: Tape<E>,
        s: <<E as Eval>::FloatSliceEval as FloatSliceEvalT<E>>::Storage,
    ) -> Self {
        let eval = E::FloatSliceEval::new_with_storage(tape.clone(), s);
        Self { tape, eval }
    }

    /// Extracts the storage from the inner [`FloatSliceEvalT`](FloatSliceEvalT)
    pub fn take(
        self,
    ) -> Option<<<E as Eval>::FloatSliceEval as FloatSliceEvalT<E>>::Storage>
    {
        self.eval.take()
    }

    /// Evaluates float slices, writing results into `out`
    pub fn eval_s(
        &mut self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        if x.len() != y.len() || x.len() != z.len() || x.len() != out.len() {
            Err(Error::MismatchedSlices)
        } else {
            self.eval.eval_s(x, y, z, out);
            Ok(())
        }
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], &mut out).unwrap();
        out[0]
    }
}

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::context::Context;

    pub fn test_give_take<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape_x = ctx.get_tape(x);
        let tape_y = ctx.get_tape(y);

        let eval = FloatSliceEval::<I>::new(tape_y.clone());
        let mut out = [0.0; 4];
        let mut t = eval.take().unwrap();

        // This is a fuzz test for icache issues
        for _ in 0..10000 {
            let mut eval =
                FloatSliceEval::<I>::new_with_storage(tape_x.clone(), t);
            eval.eval_s(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &mut out,
            )
            .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
            t = eval.take().unwrap();

            let mut eval =
                FloatSliceEval::<I>::new_with_storage(tape_y.clone(), t);
            eval.eval_s(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &mut out,
            )
            .unwrap();
            assert_eq!(out, [3.0, 2.0, 1.0, 0.0]);
            t = eval.take().unwrap();
        }
    }

    pub fn test_vectorized<I: Eval>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape(x);
        let mut eval = FloatSliceEval::<I>::new(tape);
        let mut out = [0.0; 4];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let two = ctx.constant(2.0);
        let mul = ctx.mul(y, two).unwrap();
        let tape = ctx.get_tape(mul);
        let mut eval = FloatSliceEval::<I>::new(tape);
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        eval.eval_s(
            &[0.0, 1.0, 2.0],
            &[1.0, 4.0, 8.0],
            &[0.0, 0.0, 0.0],
            &mut out[0..3],
        )
        .unwrap();
        assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

        // out is longer than inputs
        assert!(matches!(
            eval.eval_s(
                &[0.0, 1.0, 2.0],
                &[1.0, 4.0, 4.0],
                &[0.0, 0.0, 0.0],
                &mut out[0..4],
            ),
            Err(Error::MismatchedSlices)
        ));

        let mut out = [0.0; 7];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
            &[0.0; 7],
            &mut out,
        )
        .unwrap();
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
