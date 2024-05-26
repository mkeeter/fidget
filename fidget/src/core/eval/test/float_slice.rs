//! Test suite for float slice evaluators (i.e. `&[f32])`
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for such evaluators; otherwise, the module has no public exports.

use super::{build_stress_fn, test_args, CanonicalBinaryOp, CanonicalUnaryOp};
use crate::{
    context::Context,
    eval::{Function, MathFunction},
    shape::{EzShape, Shape},
    var::{Var, VarIndex},
    Error,
};
use std::collections::HashMap;

/// Helper struct to put constrains on our `Shape` object
pub struct TestFloatSlice<F>(std::marker::PhantomData<*const F>);

impl<F: Function + MathFunction> TestFloatSlice<F> {
    pub fn test_give_take() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape_x = Shape::<F>::new(&mut ctx, x).unwrap();
        let shape_y = Shape::<F>::new(&mut ctx, y).unwrap();

        // This is a fuzz test for icache issues
        let mut eval = Shape::<F>::new_float_slice_eval();
        for _ in 0..10000 {
            let tape = shape_x.ez_float_slice_tape();
            let out = eval
                .eval(
                    &tape,
                    &[0.0, 1.0, 2.0, 3.0],
                    &[3.0, 2.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0, 100.0],
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
                )
                .unwrap();
            assert_eq!(out, [3.0, 2.0, 1.0, 0.0]);
        }
    }

    pub fn test_vectorized() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let mut eval = Shape::<F>::new_float_slice_eval();
        let shape = Shape::<F>::new(&mut ctx, x).unwrap();
        let tape = shape.ez_float_slice_tape();
        let out = eval
            .eval(
                &tape,
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);

        let out = eval
            .eval(
                &tape,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let out = eval
            .eval(
                &tape,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 200.0],
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mul = ctx.mul(y, 2.0).unwrap();
        let shape = Shape::<F>::new(&mut ctx, mul).unwrap();
        let tape = shape.ez_float_slice_tape();
        let out = eval
            .eval(
                &tape,
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 2.0, 1.0, 0.0],
                &[0.0, 0.0, 0.0, 100.0],
            )
            .unwrap();
        assert_eq!(out, [6.0, 4.0, 2.0, 0.0]);

        let out = eval
            .eval(&tape, &[0.0, 1.0, 2.0], &[1.0, 4.0, 8.0], &[0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(&out[0..3], &[2.0, 8.0, 16.0]);

        let out = eval
            .eval(
                &tape,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[1.0, 4.0, 4.0, -1.0, -2.0, -3.0, 0.0],
                &[0.0; 7],
            )
            .unwrap();
        assert_eq!(out, [2.0, 8.0, 8.0, -2.0, -4.0, -6.0, 0.0]);
    }

    pub fn test_f_sin() {
        let mut ctx = Context::new();
        let a = ctx.x();
        let b = ctx.sin(a).unwrap();

        let shape = Shape::<F>::new(&mut ctx, b).unwrap();
        let mut eval = Shape::<F>::new_float_slice_eval();
        let tape = shape.ez_float_slice_tape();

        let args = [0.0, 1.0, 2.0, std::f32::consts::PI / 2.0];
        assert_eq!(
            eval.eval(&tape, &args, &[0.0; 4], &[0.0; 4],).unwrap(),
            args.map(f32::sin),
        );
    }

    pub fn test_f_var() {
        let v = Var::new();
        let mut ctx = Context::new();

        let x = ctx.x();
        let y = ctx.y();
        let v_ = ctx.var(v);

        let a = ctx.add(x, y).unwrap();
        let a = ctx.add(a, v_).unwrap();

        let s = Shape::<F>::new(&mut ctx, a).unwrap();

        let mut eval = Shape::<F>::new_float_slice_eval();
        let tape = s.ez_float_slice_tape();
        assert!(eval
            .eval(&tape, &[1.0, 2.0], &[2.0, 3.0], &[0.0, 0.0])
            .is_err());
        let mut h: HashMap<VarIndex, &[f32]> = HashMap::new();
        assert!(eval
            .eval_v(&tape, &[1.0, 2.0], &[2.0, 3.0], &[0.0, 0.0], &h)
            .is_err());
        let index = v.index().unwrap();
        h.insert(index, &[4.0, 5.0]);
        assert_eq!(
            eval.eval_v(&tape, &[1.0, 2.0], &[2.0, 3.0], &[0.0, 0.0], &h)
                .unwrap(),
            &[7.0, 10.0]
        );
        h.insert(index, &[4.0, 5.0, 6.0]);
        assert!(matches!(
            eval.eval_v(&tape, &[1.0, 2.0], &[2.0, 3.0], &[0.0, 0.0], &h),
            Err(Error::MismatchedSlices)
        ));

        // Get a new var index that isn't valid for this tape
        let v2 = Var::new();
        h.insert(index, &[4.0, 5.0]);
        h.insert(v2.index().unwrap(), &[4.0, 5.0]);
        assert!(matches!(
            eval.eval_v(&tape, &[1.0, 2.0], &[2.0, 3.0], &[0.0, 0.0], &h),
            Err(Error::BadVarSlice(..))
        ));
    }

    pub fn test_f_stress_n(depth: usize) {
        let (mut ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD register
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x = args.clone();
        let y: Vec<f32> =
            args[1..].iter().chain(&args[0..1]).cloned().collect();
        let z: Vec<f32> =
            args[2..].iter().chain(&args[0..2]).cloned().collect();

        let shape = Shape::<F>::new(&mut ctx, node).unwrap();
        let mut eval = Shape::<F>::new_float_slice_eval();
        let tape = shape.ez_float_slice_tape();

        let out = eval.eval(&tape, &x, &y, &z).unwrap();

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

        // Compare against the VmShape evaluator as a baseline.  It's possible
        // that S is also a VmShape, but this comparison isn't particularly
        // expensive, so we'll do it regardless.
        use crate::vm::VmShape;
        let shape = VmShape::new(&mut ctx, node).unwrap();
        let mut eval = VmShape::new_float_slice_eval();
        let tape = shape.ez_float_slice_tape();

        let cmp = eval.eval(&tape, &x, &y, &z).unwrap();
        for (i, (a, b)) in out.iter().zip(cmp.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err < 1e-6,
                "mismatch at index {i} ({}, {}, {}): {a} != {b} [{err}], {}",
                x[i],
                y[i],
                z[i],
                depth,
            );
        }
    }

    pub fn test_f_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32] {
            Self::test_f_stress_n(n);
        }
    }

    pub fn test_unary<C: CanonicalUnaryOp>() {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        for (i, v) in [ctx.x(), ctx.y(), ctx.z()].into_iter().enumerate() {
            let node = C::build(&mut ctx, v);

            let shape = Shape::<F>::new(&mut ctx, node).unwrap();
            let mut eval = Shape::<F>::new_float_slice_eval();
            let tape = shape.ez_float_slice_tape();

            let out = match i {
                0 => eval.eval(&tape, &args, &zero, &zero),
                1 => eval.eval(&tape, &zero, &args, &zero),
                2 => eval.eval(&tape, &zero, &zero, &args),
                _ => unreachable!(),
            }
            .unwrap();
            for (a, &o) in args.iter().zip(out.iter()) {
                let v = C::eval_f32(*a);
                let err = (v - o).abs();
                assert!(
                    (o == v) || err < 1e-6 || (v.is_nan() && o.is_nan()),
                    "mismatch in '{}' at {a}: {v} != {o} ({err})",
                    C::NAME,
                )
            }
        }
    }

    pub fn compare_float_results<C: CanonicalBinaryOp>(
        lhs: &[f32],
        rhs: &[f32],
        out: &[f32],
        g: impl Fn(f32, f32) -> f32,
        name: &str,
    ) {
        for ((a, b), &o) in lhs.iter().zip(rhs).zip(out.iter()) {
            let v = g(*a, *b);
            let err = (v - o).abs();
            assert!(
                (o == v)
                    || C::discontinuous_at(*a, *b)
                    || err < 1e-6
                    || (v.is_nan() && o.is_nan()),
                "mismatch in '{name}' at {a} {b}: {v} != {o} ({err})"
            )
        }
    }

    pub fn test_binary_reg_reg<C: CanonicalBinaryOp>() {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];
        let name = format!("{}(reg, reg)", C::NAME);
        for rot in 0..args.len() {
            let mut rgsa = args.clone();
            rgsa.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for (j, &u) in inputs.iter().enumerate() {
                    let node = C::build(&mut ctx, v, u);

                    let shape = Shape::<F>::new(&mut ctx, node).unwrap();
                    let mut eval = Shape::<F>::new_float_slice_eval();
                    let tape = shape.ez_float_slice_tape();

                    let out = match (i, j) {
                        (0, 0) => eval.eval(&tape, &args, &zero, &zero),
                        (0, 1) => eval.eval(&tape, &args, &rgsa, &zero),
                        (0, 2) => eval.eval(&tape, &args, &zero, &rgsa),
                        (1, 0) => eval.eval(&tape, &rgsa, &args, &zero),
                        (1, 1) => eval.eval(&tape, &zero, &args, &zero),
                        (1, 2) => eval.eval(&tape, &zero, &args, &rgsa),
                        (2, 0) => eval.eval(&tape, &rgsa, &zero, &args),
                        (2, 1) => eval.eval(&tape, &zero, &rgsa, &args),
                        (2, 2) => eval.eval(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let rhs = if i == j { &args } else { &rgsa };
                    Self::compare_float_results::<C>(
                        &args,
                        rhs,
                        out,
                        C::eval_reg_reg_f32,
                        &name,
                    );
                }
            }
        }
    }

    fn test_binary_reg_imm<C: CanonicalBinaryOp>() {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{}(reg, imm)", C::NAME);
        for rot in 0..args.len() {
            let mut args = args.clone();
            args.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for rhs in args.iter() {
                    let c = ctx.constant(*rhs as f64);
                    let node = C::build(&mut ctx, v, c);

                    let shape = Shape::<F>::new(&mut ctx, node).unwrap();
                    let mut eval = Shape::<F>::new_float_slice_eval();
                    let tape = shape.ez_float_slice_tape();

                    let out = match i {
                        0 => eval.eval(&tape, &args, &zero, &zero),
                        1 => eval.eval(&tape, &zero, &args, &zero),
                        2 => eval.eval(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let rhs = vec![*rhs; out.len()];
                    Self::compare_float_results::<C>(
                        &args,
                        &rhs,
                        out,
                        C::eval_reg_imm_f32,
                        &name,
                    );
                }
            }
        }
    }

    fn test_binary_imm_reg<C: CanonicalBinaryOp>() {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{}(imm, reg)", C::NAME);
        for rot in 0..args.len() {
            let mut args = args.clone();
            args.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for lhs in args.iter() {
                    let c = ctx.constant(*lhs as f64);
                    let node = C::build(&mut ctx, c, v);

                    let shape = Shape::<F>::new(&mut ctx, node).unwrap();
                    let mut eval = Shape::<F>::new_float_slice_eval();
                    let tape = shape.ez_float_slice_tape();

                    let out = match i {
                        0 => eval.eval(&tape, &args, &zero, &zero),
                        1 => eval.eval(&tape, &zero, &args, &zero),
                        2 => eval.eval(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let lhs = vec![*lhs; out.len()];
                    Self::compare_float_results::<C>(
                        &lhs,
                        &args,
                        out,
                        C::eval_imm_reg_f32,
                        &name,
                    );
                }
            }
        }
    }

    pub fn test_binary<C: CanonicalBinaryOp>() {
        Self::test_binary_reg_reg::<C>();
        Self::test_binary_reg_imm::<C>();
        Self::test_binary_imm_reg::<C>();
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
        $crate::float_slice_test!(test_f_sin, $t);
        $crate::float_slice_test!(test_f_var, $t);
        $crate::float_slice_test!(test_f_stress, $t);

        mod f_unary {
            use super::*;
            $crate::all_unary_tests!(
                $crate::eval::test::float_slice::TestFloatSlice::<$t>
            );
        }

        mod f_binary {
            use super::*;
            $crate::all_binary_tests!(
                $crate::eval::test::float_slice::TestFloatSlice::<$t>
            );
        }
    };
}
