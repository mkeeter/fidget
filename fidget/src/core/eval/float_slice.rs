//! Test suite for float slice evaluators (i.e. `&[f32])`
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for such evaluators; otherwise, the module has no public exports.

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use crate::{
        context::{Context, Node},
        eval::{BulkEvaluator, EzShape, Shape, ShapeVars, Vars},
    };

    /// Helper struct to put constrains on our `Shape` object
    pub struct TestFloatSlice<S>(std::marker::PhantomData<*const S>);

    impl<S> TestFloatSlice<S>
    where
        for<'a> S: Shape + TryFrom<(&'a Context, Node)> + ShapeVars,
        for<'a> <S as TryFrom<(&'a Context, Node)>>::Error: std::fmt::Debug,
    {
        pub fn test_give_take() {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();

            let shape_x = S::try_from((&ctx, x)).unwrap();
            let shape_y = S::try_from((&ctx, y)).unwrap();

            // This is a fuzz test for icache issues
            let mut eval = S::new_float_slice_eval();
            for _ in 0..10000 {
                let tape = shape_x.ez_float_slice_tape();
                let out = eval
                    .eval(
                        &tape,
                        &[0.0, 1.0, 2.0, 3.0],
                        &[3.0, 2.0, 1.0, 0.0],
                        &[0.0, 0.0, 0.0, 100.0],
                        &[],
                    )
                    .unwrap();
                assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

                // TODO: reuse tape data here

                let tape = shape_y.ez_float_slice_tape();
                let out = eval
                    .eval(
                        &tape,
                        &[0.0, 1.0, 2.0, 3.0],
                        &[3.0, 2.0, 1.0, 0.0],
                        &[0.0, 0.0, 0.0, 100.0],
                        &[],
                    )
                    .unwrap();
                assert_eq!(out, [3.0, 2.0, 1.0, 0.0]);
            }
        }

        pub fn test_vectorized() {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();

            let mut eval = S::new_float_slice_eval();
            let shape = S::try_from((&ctx, x)).unwrap();
            let tape = shape.ez_float_slice_tape();
            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                    &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 200.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

            let mul = ctx.mul(y, 2.0).unwrap();
            let shape = S::try_from((&ctx, mul)).unwrap();
            let tape = shape.ez_float_slice_tape();
            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0],
                    &[1.0, 4.0, 8.0],
                    &[0.0, 0.0, 0.0],
                    &[],
                )
                .unwrap();
            assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
                    &[0.0; 7],
                    &[],
                )
                .unwrap();
            assert_eq!(out, [2.0, 8.0, 8.0, -2.0, -4.0, -6.0, 0.0]);
        }

        pub fn test_f_var() {
            let mut ctx = Context::new();
            let a = ctx.var("a").unwrap();
            let b = ctx.var("b").unwrap();
            let sum = ctx.add(a, 1.0).unwrap();
            let min = ctx.div(sum, b).unwrap();

            let shape = S::try_from((&ctx, min)).unwrap();
            let mut eval = S::new_float_slice_eval();
            let tape = shape.ez_float_slice_tape();
            let mut vars = Vars::new(shape.vars());

            assert_eq!(
                eval.eval(
                    &tape,
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
                    &tape,
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
                    &tape,
                    &[0.0],
                    &[0.0],
                    &[0.0],
                    vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
                )
                .unwrap()[0],
                0.5,
            );
        }
    }

    #[macro_export]
    macro_rules! float_slice_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::float_slice::eval_tests::TestFloatSlice::<$t>::$i(
                )
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
