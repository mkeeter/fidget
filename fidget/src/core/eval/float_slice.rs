//! Test suite for float slice evaluators (i.e. `&[f32])`
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for such evaluators; otherwise, the module has no public exports.

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use crate::{
        context::Context,
        eval::{BulkEvaluator, MathShape, ShapeFloatSliceEval, Vars},
    };

    pub fn test_give_take<S: ShapeFloatSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape_x = S::new(&ctx, x).unwrap();
        let tape_y = S::new(&ctx, y).unwrap();

        // This is a fuzz test for icache issues
        let mut eval = S::Eval::new();
        for _ in 0..10000 {
            let t = tape_x.tape();
            let out = eval
                .eval(
                    &t,
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
            t = eval.take().unwrap();

            let t = tape_y.tape();
            let out = eval
                .eval(
                    &t,
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

    pub fn test_vectorized<S: ShapeFloatSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let mut eval = S::Eval::new();
        let tape = S::new(&ctx, x).unwrap();
        let t = tape.tape();
        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 200.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mul = ctx.mul(y, 2.0).unwrap();
        let tape = S::new(&ctx, mul).unwrap();
        let t = tape.tape();
        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
                &[],
            )
            .unwrap();
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0],
                &[1.0, 4.0, 8.0],
                &[0.0, 0.0, 0.0],
                &[],
            )
            .unwrap();
        assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

        let out = eval
            .eval(
                &t,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
                &[0.0; 7],
                &[],
            )
            .unwrap();
        assert_eq!(out, [2.0, 8.0, 8.0, -2.0, -4.0, -6.0, 0.0]);
    }

    pub fn test_f_var<S: ShapeFloatSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let eval = S::Eval::new();
        let t = tape.tape();
        let mut vars = Vars::new(&tape);

        assert_eq!(
            eval.eval(
                &t,
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
                &t,
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
                &t,
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
