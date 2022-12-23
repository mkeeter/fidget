//! Float slice evaluation (i.e. `&[f32]`)
use crate::{
    eval::{Family, Tape},
    Error,
};

/// Function handle for evaluation of many points simultaneously.
pub trait FloatSliceEvalT<R> {
    /// Storage used by the evaluator, provided to minimize allocation churn
    type Storage: Default;

    fn new(tape: &Tape<R>) -> Self;

    /// Constructs the `FloatSliceT`, giving it a chance to reuse storage
    ///
    /// In the default implementation, `_storage` is ignored; override this
    /// function if it would be useful.
    ///
    /// The incoming `Storage` is consumed, though it may not necessarily be
    /// used to construct the new tape (e.g. if it's a memory-mapped region and
    /// is too small).
    fn new_with_storage(tape: &Tape<R>, _storage: Self::Storage) -> Self
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
    /// This function may assume that the `x`, `y`, `z`, and `out` slices are of
    /// equal length and panic otherwise; higher-level calls (e.g.
    /// [`FloatSliceEval::eval_s`](`FloatSliceEval::eval_s`)) should maintain
    /// that invariant.
    ///
    /// This function may also assume that `vars` is correctly sized for the
    /// number of variables in the tape.
    fn eval_s(
        &mut self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
        out: &mut [f32],
    );
}

/// Evaluator for float slices, parameterized by evaluator family
pub struct FloatSliceEval<E: Family> {
    #[allow(dead_code)]
    tape: Tape<E>,
    eval: E::FloatSliceEval,
}

impl<E: Family> FloatSliceEval<E> {
    pub fn new(tape: Tape<E>) -> Self {
        let eval = E::FloatSliceEval::new(&tape);
        Self { tape, eval }
    }

    /// Builds a new [`FloatSliceEval`](Self), reusing storage to minimize churn
    pub fn new_with_storage(
        tape: Tape<E>,
        s: FloatSliceEvalStorage<E>,
    ) -> Self {
        let eval = E::FloatSliceEval::new_with_storage(&tape, s.inner);
        Self { tape, eval }
    }

    /// Extracts the storage from the inner [`FloatSliceEvalT`](FloatSliceEvalT)
    pub fn take(self) -> Option<FloatSliceEvalStorage<E>> {
        self.eval
            .take()
            .map(|inner| FloatSliceEvalStorage { inner })
    }

    /// Evaluates float slices, writing results into `out`
    pub fn eval_s(
        &mut self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        if x.len() != y.len() || x.len() != z.len() || x.len() != out.len() {
            Err(Error::MismatchedSlices)
        } else if vars.len() != self.tape.var_count() {
            Err(Error::BadVarSlice(vars.len(), self.tape.var_count()))
        } else {
            self.eval.eval_s(x, y, z, vars, out);
            Ok(())
        }
    }
    pub fn eval_f(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        vars: &[f32],
    ) -> Result<f32, Error> {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], vars, &mut out)?;
        Ok(out[0])
    }
}

/// Helper `struct` to reuse storage from an [`FloatSliceEval`](FloatSliceEval)
pub struct FloatSliceEvalStorage<E: Family> {
    inner: <<E as Family>::FloatSliceEval as FloatSliceEvalT<E>>::Storage,
}

impl<E: Family> Default for FloatSliceEvalStorage<E> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{Eval, Vars},
    };

    pub fn test_give_take<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape_x = ctx.get_tape(x).unwrap();
        let tape_y = ctx.get_tape(y).unwrap();

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
                &[],
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
                &[],
                &mut out,
            )
            .unwrap();
            assert_eq!(out, [3.0, 2.0, 1.0, 0.0]);
            t = eval.take().unwrap();
        }
    }

    pub fn test_vectorized<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape(x).unwrap();
        let mut eval = FloatSliceEval::<I>::new(tape);
        let mut out = [0.0; 4];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let mul = ctx.mul(y, 2.0).unwrap();
        let tape = ctx.get_tape(mul).unwrap();
        let mut eval = FloatSliceEval::<I>::new(tape);
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 100.0],
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        eval.eval_s(
            &[0.0, 1.0, 2.0],
            &[1.0, 4.0, 8.0],
            &[0.0, 0.0, 0.0],
            &[],
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
                &[],
                &mut out[0..4],
            ),
            Err(Error::MismatchedSlices)
        ));

        let mut out = [0.0; 7];
        eval.eval_s(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
            &[0.0; 7],
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [2.0, 8.0, 8.0, -2.0, -4.0, -6.0, 0.0]);
    }

    pub fn test_f_var<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = ctx.get_tape(min).unwrap();
        let mut vars = Vars::new(&tape);
        let mut eval = I::new_float_slice_evaluator(tape);

        assert_eq!(
            eval.eval_f(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap(),
            2.0
        );
        assert_eq!(
            eval.eval_f(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 3.0), ("b", 2.0)].into_iter())
            )
            .unwrap(),
            2.0
        );
        assert_eq!(
            eval.eval_f(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
            )
            .unwrap(),
            0.5,
        );
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
            $crate::float_slice_test!(test_f_var, $t);
        };
    }
}
