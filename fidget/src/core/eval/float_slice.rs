//! Float slice evaluation (i.e. `&[f32]`)
use crate::eval::{
    bulk::{BulkEval, BulkEvalData, BulkEvaluator},
    EvaluatorStorage, Family,
};

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for many points, returning a bunch of `f32`'s
pub type FloatSliceEval<F> = BulkEval<f32, <F as Family>::FloatSliceEval, F>;

/// Scratch data used by an bulk float evaluator from a particular family `F`
pub type FloatSliceEvalData<F> = BulkEvalData<
    <<F as Family>::FloatSliceEval as BulkEvaluator<f32, F>>::Data,
    f32,
    F,
>;

/// Immutable data used by an bulk float evaluator from a particular family `F`
pub type FloatSliceEvalStorage<F> =
    <<F as Family>::FloatSliceEval as EvaluatorStorage<F>>::Storage;

////////////////////////////////////////////////////////////////////////////////

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{Tape, Vars},
    };

    pub fn test_give_take<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape_x = Tape::new(&ctx, x).unwrap();
        let tape_y = Tape::new(&ctx, y).unwrap();

        let eval = FloatSliceEval::<I>::new(&tape_y);
        let mut t = eval.take().unwrap();

        // This is a fuzz test for icache issues
        for _ in 0..10000 {
            let eval = FloatSliceEval::<I>::new_with_storage(&tape_x, t);
            let out = eval
                .eval(
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
            t = eval.take().unwrap();

            let eval = FloatSliceEval::<I>::new_with_storage(&tape_y, t);
            let out = eval
                .eval(
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
                    &[],
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

        let tape = Tape::new(&ctx, x).unwrap();
        let eval = FloatSliceEval::<I>::new(&tape);
        let out = eval
            .eval(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let out = eval
            .eval(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let out = eval
            .eval(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 200.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mul = ctx.mul(y, 2.0).unwrap();
        let tape = Tape::new(&ctx, mul).unwrap();
        let eval = FloatSliceEval::<I>::new(&tape);
        let out = eval
            .eval(
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        let out = eval
            .eval(&[0.0, 1.0, 2.0], &[1.0, 4.0, 8.0], &[0.0, 0.0, 0.0], &[])
            .unwrap();
        assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

        let out = eval
            .eval(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
                &[0.0; 7],
                &[],
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
        let tape = Tape::<I>::new(&ctx, min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = tape.new_float_slice_evaluator();

        assert_eq!(
            eval.eval(
                &[0.0],
                &[0.0],
                &[0.0],
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap()[0],
            2.0
        );
        assert_eq!(
            eval.eval(
                &[0.0],
                &[0.0],
                &[0.0],
                vars.bind([("a", 3.0), ("b", 2.0)].into_iter())
            )
            .unwrap()[0],
            2.0
        );
        assert_eq!(
            eval.eval(
                &[0.0],
                &[0.0],
                &[0.0],
                vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
            )
            .unwrap()[0],
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
