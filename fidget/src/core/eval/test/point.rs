//! Test suite for single-point tracing evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for point evaluators; otherwise, the module has no public exports.
use super::{build_stress_fn, test_args, CanonicalBinaryOp, CanonicalUnaryOp};
use crate::{
    context::Context,
    eval::{EzShape, MathShape, Shape, ShapeVars, TracingEvaluator, Vars},
    vm::Choice,
};

/// Helper struct to put constrains on our `Shape` object
pub struct TestPoint<S>(std::marker::PhantomData<*const S>);
impl<S> TestPoint<S>
where
    S: Shape + MathShape + ShapeVars,
    <S as Shape>::Trace: AsRef<[Choice]>,
    <S as Shape>::Trace: From<Vec<Choice>>,
{
    pub fn test_constant() {
        let mut ctx = Context::new();
        let p = ctx.constant(1.5);
        let shape = S::new(&ctx, p).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_constant_push() {
        let mut ctx = Context::new();
        let a = ctx.constant(1.5);
        let x = ctx.x();
        let min = ctx.min(a, x).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.5);

        let next = shape.ez_simplify(trace.unwrap()).unwrap();
        assert_eq!(next.size(), 1);

        let tape = next.ez_point_tape();
        assert_eq!(eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
        assert_eq!(eval.eval(&tape, 1.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_circle() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let circle = ctx.sub(radius, 1.0).unwrap();

        let shape = S::new(&ctx, circle).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap().0, -1.0);
        assert_eq!(eval.eval(&tape, 1.0, 0.0, 0.0, &[]).unwrap().0, 0.0);
    }

    pub fn test_p_min()
    where
        <S as Shape>::Trace: AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        let (r, trace) = eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, 0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, 0.0, f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());
    }

    pub fn test_p_max()
    where
        <S as Shape>::Trace: AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let shape = S::new(&ctx, max).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();

        let (r, trace) = eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, 0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 2.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, 0.0, f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());
    }

    pub fn test_p_sin()
    where
        <S as Shape>::Trace: AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();

        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();

        for x in [0.0, 1.0, 2.0] {
            let (r, trace) = eval.eval(&tape, x, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, x, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, x, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());
        }

        let y = ctx.y();
        let s = ctx.add(s, y).unwrap();
        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_point_tape();

        for (x, y) in [(0.0, 1.0), (1.0, 3.0), (2.0, 8.0)] {
            let (r, trace) = eval.eval(&tape, x, y, 0.0, &[]).unwrap();
            assert_eq!(r, x.sin() + y);
            assert!(trace.is_none());
        }
    }

    pub fn basic_interpreter() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sum = ctx.add(x, 1.0).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, 1.0, 3.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, 3.0, 3.5, 0.0, &[]).unwrap().0, 3.5);
    }

    pub fn test_push()
    where
        <S as Shape>::Trace: AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let next = shape.ez_simplify(&vec![Choice::Left].into()).unwrap();
        let tape = next.ez_point_tape();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 3.0);

        let next = shape.ez_simplify(&vec![Choice::Right].into()).unwrap();
        let tape = next.ez_point_tape();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let min = ctx.min(x, 1.0).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);

        let next = shape.ez_simplify(&vec![Choice::Left].into()).unwrap();
        let tape = next.ez_point_tape();
        assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 3.0);

        let next = shape.ez_simplify(&vec![Choice::Right].into()).unwrap();
        let tape = next.ez_point_tape();
        assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);
    }

    pub fn test_basic() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape = S::new(&ctx, x).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, 3.0, 4.0, 0.0, &[]).unwrap().0, 3.0);

        let shape = S::new(&ctx, y).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, 3.0, 4.0, 0.0, &[]).unwrap().0, 4.0);

        let y2 = ctx.mul(y, 2.5).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let shape = S::new(&ctx, sum).unwrap();
        let tape = shape.ez_point_tape();
        let mut eval = S::new_point_eval();
        assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 6.0);
    }

    pub fn test_var() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let shape = S::new(&ctx, a).unwrap();
        let tape = shape.ez_point_tape();
        let mut vars = Vars::new(shape.vars());
        let mut eval = S::new_point_eval();
        assert_eq!(
            eval.eval(
                &tape,
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 5.0)].into_iter())
            )
            .unwrap()
            .0,
            5.0
        );
        assert_eq!(
            eval.eval(
                &tape,
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 1.0)].into_iter())
            )
            .unwrap()
            .0,
            1.0
        );

        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_point_tape();
        let mut vars = Vars::new(shape.vars());
        let mut eval = S::new_point_eval();

        assert_eq!(
            eval.eval(
                &tape,
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap()
            .0,
            2.0
        );
        assert_eq!(
            eval.eval(
                &tape,
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 3.0), ("b", 2.0)].into_iter())
            )
            .unwrap()
            .0,
            2.0
        );
        assert_eq!(
            eval.eval(
                &tape,
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
            )
            .unwrap()
            .0,
            0.5
        );
    }

    pub fn test_p_stress_n(depth: usize) {
        let (ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD register
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x: Vec<_> = args.clone();
        let y: Vec<_> = x[1..].iter().chain(&x[0..1]).cloned().collect();
        let z: Vec<_> = x[2..].iter().chain(&x[0..2]).cloned().collect();

        let shape = S::new(&ctx, node).unwrap();
        let mut eval = S::new_point_eval();
        let tape = shape.ez_point_tape();

        let mut out = vec![];
        for i in 0..args.len() {
            out.push(eval.eval(&tape, x[i], y[i], z[i], &[]).unwrap().0);
        }

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
        let shape = VmShape::new(&ctx, node).unwrap();
        let mut eval = VmShape::new_point_eval();
        let tape = shape.ez_point_tape();

        let mut cmp = vec![];
        for i in 0..args.len() {
            cmp.push(eval.eval(&tape, x[i], y[i], z[i], &[]).unwrap().0);
        }

        for (a, b) in out.iter().zip(cmp.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    pub fn test_p_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32] {
            Self::test_p_stress_n(n);
        }
    }

    pub fn test_unary<C: CanonicalUnaryOp>() {
        // Pick a bunch of arguments, some of which are spicy
        let args = test_args();

        let mut ctx = Context::new();
        for (i, v) in [ctx.x(), ctx.y(), ctx.z()].into_iter().enumerate() {
            let node = C::build(&mut ctx, v);

            let shape = S::new(&ctx, node).unwrap();
            let mut eval = S::new_point_eval();
            let tape = shape.ez_point_tape();

            for &a in args.iter() {
                let (o, trace) = match i {
                    0 => eval.eval(&tape, a, 0.0, 0.0, &[]),
                    1 => eval.eval(&tape, 0.0, a, 0.0, &[]),
                    2 => eval.eval(&tape, 0.0, 0.0, a, &[]),
                    _ => unreachable!(),
                }
                .unwrap();
                assert!(trace.is_none());
                let v = C::eval_f32(a);
                let err = (v - o).abs();
                assert!(
                    (o == v) || err < 1e-6 || (v.is_nan() && o.is_nan()),
                    "mismatch in '{}' at {a}: {v} != {o} ({err})",
                    C::NAME,
                )
            }
        }
    }

    fn compare_point_results(
        lhs: f32,
        rhs: f32,
        out: f32,
        g: impl Fn(f32, f32) -> f32,
        name: &str,
    ) {
        let value = g(lhs, rhs);
        let err = (value - out).abs();
        assert!(
            (out == value) || err < 1e-6 || value.is_nan() && out.is_nan(),
            "mismatch in '{name}' at ({lhs}, {rhs}): \
                            {value} != {out} ({err})"
        )
    }

    pub fn test_binary_reg_reg<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{}(reg, reg)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    for (j, &v) in xyz.iter().enumerate() {
                        let node = C::build(&mut ctx, u, v);

                        let shape = S::new(&ctx, node).unwrap();
                        let mut eval = S::new_point_eval();
                        let tape = shape.ez_point_tape();

                        let (out, _trace) = match (i, j) {
                            (0, 0) => eval.eval(&tape, lhs, 0.0, 0.0, &[]),
                            (0, 1) => eval.eval(&tape, lhs, rhs, 0.0, &[]),
                            (0, 2) => eval.eval(&tape, lhs, 0.0, rhs, &[]),
                            (1, 0) => eval.eval(&tape, rhs, lhs, 0.0, &[]),
                            (1, 1) => eval.eval(&tape, 0.0, lhs, 0.0, &[]),
                            (1, 2) => eval.eval(&tape, 0.0, lhs, rhs, &[]),
                            (2, 0) => eval.eval(&tape, rhs, 0.0, lhs, &[]),
                            (2, 1) => eval.eval(&tape, 0.0, rhs, lhs, &[]),
                            (2, 2) => eval.eval(&tape, 0.0, 0.0, lhs, &[]),
                            _ => unreachable!(),
                        }
                        .unwrap();

                        let rhs = if i == j { lhs } else { rhs };
                        Self::compare_point_results(
                            lhs,
                            rhs,
                            out,
                            C::eval_reg_reg_f32,
                            &name,
                        );
                    }
                }
            }
        }
    }

    pub fn test_binary_reg_imm<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{}(reg, imm)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    let c = ctx.constant(rhs as f64);
                    let node = C::build(&mut ctx, u, c);

                    let shape = S::new(&ctx, node).unwrap();
                    let mut eval = S::new_point_eval();
                    let tape = shape.ez_point_tape();

                    let (out, _trace) = match i {
                        0 => eval.eval(&tape, lhs, 0.0, 0.0, &[]),
                        1 => eval.eval(&tape, 0.0, lhs, 0.0, &[]),
                        2 => eval.eval(&tape, 0.0, 0.0, lhs, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    Self::compare_point_results(
                        lhs,
                        rhs,
                        out,
                        C::eval_reg_imm_f32,
                        &name,
                    );
                }
            }
        }
    }

    pub fn test_binary_imm_reg<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{}(imm, reg)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    let c = ctx.constant(lhs as f64);
                    let node = C::build(&mut ctx, c, u);

                    let shape = S::new(&ctx, node).unwrap();
                    let mut eval = S::new_point_eval();
                    let tape = shape.ez_point_tape();

                    let (out, _trace) = match i {
                        0 => eval.eval(&tape, rhs, 0.0, 0.0, &[]),
                        1 => eval.eval(&tape, 0.0, rhs, 0.0, &[]),
                        2 => eval.eval(&tape, 0.0, 0.0, rhs, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    Self::compare_point_results(
                        lhs,
                        rhs,
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

    pub fn test_p_binary_ops() {
        use super::canonical::*;

        Self::test_binary::<add>();
        Self::test_binary::<sub>();
        Self::test_binary::<mul>();
        Self::test_binary::<div>();
        Self::test_binary::<min>();
        Self::test_binary::<max>();
    }

    pub fn test_p_unary_ops() {
        use super::canonical::*;

        Self::test_unary::<neg>();
        Self::test_unary::<recip>();
        Self::test_unary::<abs>();
        Self::test_unary::<sin>();
        Self::test_unary::<cos>();
        Self::test_unary::<tan>();
        Self::test_unary::<asin>();
        Self::test_unary::<acos>();
        Self::test_unary::<atan>();
        Self::test_unary::<exp>();
        Self::test_unary::<ln>();
        Self::test_unary::<square>();
        Self::test_unary::<sqrt>();
    }
}

#[macro_export]
macro_rules! point_test {
    ($i:ident, $t:ty) => {
        #[test]
        fn $i() {
            $crate::eval::test::point::TestPoint::<$t>::$i()
        }
    };
}

#[macro_export]
macro_rules! point_tests {
    ($t:ty) => {
        $crate::point_test!(test_constant, $t);
        $crate::point_test!(test_constant_push, $t);
        $crate::point_test!(test_circle, $t);
        $crate::point_test!(test_p_max, $t);
        $crate::point_test!(test_p_min, $t);
        $crate::point_test!(test_p_sin, $t);
        $crate::point_test!(basic_interpreter, $t);
        $crate::point_test!(test_push, $t);
        $crate::point_test!(test_var, $t);
        $crate::point_test!(test_basic, $t);
        $crate::point_test!(test_p_stress, $t);
        $crate::point_test!(test_p_unary_ops, $t);
        $crate::point_test!(test_p_binary_ops, $t);
    };
}
