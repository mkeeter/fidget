//! Test suite for float slice evaluators (i.e. `&[f32])`
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for such evaluators; otherwise, the module has no public exports.

use super::build_stress_fn;
use crate::{
    context::Context,
    eval::{BulkEvaluator, EzShape, MathShape, Shape, ShapeVars, Vars},
};

/// Helper struct to put constrains on our `Shape` object
pub struct TestFloatSlice<S>(std::marker::PhantomData<*const S>);

impl<S> TestFloatSlice<S>
where
    S: Shape + MathShape + ShapeVars,
{
    pub fn test_give_take() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape_x = S::new(&ctx, x).unwrap();
        let shape_y = S::new(&ctx, y).unwrap();

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
        let shape = S::new(&ctx, x).unwrap();
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
        let shape = S::new(&ctx, mul).unwrap();
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

        let shape = S::new(&ctx, min).unwrap();
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

    pub fn test_f_sin() {
        let mut ctx = Context::new();
        let a = ctx.x();
        let b = ctx.sin(a).unwrap();

        let shape = S::new(&ctx, b).unwrap();
        let mut eval = S::new_float_slice_eval();
        let tape = shape.ez_float_slice_tape();
        let mut vars = Vars::new(shape.vars());

        let args = [0.0, 1.0, 2.0, std::f32::consts::PI / 2.0];
        assert_eq!(
            eval.eval(
                &tape,
                &args,
                &[0.0; 4],
                &[0.0; 4],
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap(),
            args.map(f32::sin),
        );
    }

    pub fn test_f_stress_n(depth: usize) {
        let (ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD register
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x = args.clone();
        let y: Vec<f32> =
            args[1..].iter().chain(&args[0..1]).cloned().collect();
        let z: Vec<f32> =
            args[2..].iter().chain(&args[0..2]).cloned().collect();

        let shape = S::new(&ctx, node).unwrap();
        let mut eval = S::new_float_slice_eval();
        let tape = shape.ez_float_slice_tape();

        let out = eval.eval(&tape, &x, &y, &z, &[]).unwrap();

        for (i, v) in out.iter().cloned().enumerate() {
            let q = ctx
                .eval_xyz(node, x[i] as f64, y[i] as f64, z[i] as f64)
                .unwrap();
            let err = (v as f64 - q).abs();
            assert!(
                err < 1e-2, // generous error bounds, for the 512-op case
                "mismatch at index {i} ({}, {}, {}): {v} != {q} [{err}], {}",
                x[i],
                y[i],
                z[i],
                depth,
            );
        }
    }

    pub fn test_f_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32, 512] {
            Self::test_f_stress_n(n);
        }
    }
}

#[macro_export]
macro_rules! float_slice_test {
    ($i:ident, $t:ty) => {
        #[test]
        fn $i() {
            $crate::eval::test::float_slice::TestFloatSlice::<$t>::$i()
        }
    };
}

#[macro_export]
macro_rules! float_slice_tests {
    ($t:ty) => {
        $crate::float_slice_test!(test_give_take, $t);
        $crate::float_slice_test!(test_vectorized, $t);
        $crate::float_slice_test!(test_f_var, $t);
        $crate::float_slice_test!(test_f_sin, $t);
        $crate::float_slice_test!(test_f_stress, $t);
    };
}
