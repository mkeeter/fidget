//! Test suite for single-point tracing evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for point evaluators; otherwise, the module has no public exports.
use super::{
    bind_xy, bind_xyz, build_stress_fn, test_args, CanonicalBinaryOp,
    CanonicalUnaryOp,
};
use crate::{
    context::Context,
    eval::{Function, MathFunction, Tape, TracingEvaluator},
    shape::{EzShape, Shape},
    var::Var,
    vm::Choice,
};
use std::collections::HashMap;

/// Helper struct to put constrains on our `Shape` object
pub struct TestPoint<F>(std::marker::PhantomData<*const F>);
impl<F> TestPoint<F>
where
    F: Function + MathFunction,
    <F as Function>::Trace: AsRef<[Choice]>,
    <F as Function>::Trace: From<Vec<Choice>>,
{
    pub fn test_constant() {
        let mut ctx = Context::new();
        let p = ctx.constant(1.5);
        let shape = F::new(&ctx, p).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &[]).unwrap().0, 1.5);
    }

    pub fn test_constant_push() {
        let mut ctx = Context::new();
        let min = ctx.min(1.5, Var::X).unwrap();
        let shape = F::new(&ctx, min).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        let (r, trace) = eval.eval(&tape, &[2.0]).unwrap();
        assert_eq!(r, 1.5);

        let next = shape
            .simplify(
                trace.unwrap(),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        assert_eq!(next.size(), 2); // constant, output

        let tape = next.point_tape(Default::default());
        assert_eq!(eval.eval(&tape, &[2.0]).unwrap().0, 1.5);
        assert_eq!(eval.eval(&tape, &[1.0]).unwrap().0, 1.5);
        assert!(eval.eval(&tape, &[]).is_err());
    }

    pub fn test_circle() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let circle = ctx.sub(radius, 1.0).unwrap();

        let shape = F::new(&ctx, circle).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &[0.0, 0.0]).unwrap().0, -1.0);
        assert_eq!(eval.eval(&tape, &[1.0, 0.0]).unwrap().0, 0.0);
    }

    pub fn test_p_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = F::new(&ctx, min).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();

        let (r, trace) = eval.eval(&tape, &vs(0.0, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, &vs(0.0, 1.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(2.0, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(f32::NAN, 0.0)).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, &vs(0.0, f32::NAN)).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());
    }

    pub fn test_p_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let shape = F::new(&ctx, max).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();

        let (r, trace) = eval.eval(&tape, &vs(0.0, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, &vs(0.0, 1.0)).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(2.0, 0.0)).unwrap();
        assert_eq!(r, 2.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(f32::NAN, 0.0)).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());

        let (r, trace) = eval.eval(&tape, &vs(0.0, f32::NAN)).unwrap();
        assert!(r.is_nan());
        assert!(trace.is_none());
    }

    pub fn test_p_and() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let v = ctx.and(x, y).unwrap();

        let shape = F::new(&ctx, v).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();

        let (r, trace) = eval.eval(&tape, &vs(0.0, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(0.0, 1.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(0.0, f32::NAN)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(0.1, 1.0)).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(0.1, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(f32::NAN, 1.2)).unwrap();
        assert_eq!(r, 1.2);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);
    }

    pub fn test_p_or() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let v = ctx.or(x, y).unwrap();

        let shape = F::new(&ctx, v).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);

        let mut eval = F::new_point_eval();
        let (r, trace) = eval.eval(&tape, &vs(0.0, 0.0)).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(0.0, 1.0)).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(0.0, f32::NAN)).unwrap();
        assert!(r.is_nan());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) = eval.eval(&tape, &vs(0.1, 1.0)).unwrap();
        assert_eq!(r, 0.1);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(0.1, 0.0)).unwrap();
        assert_eq!(r, 0.1);
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) = eval.eval(&tape, &vs(f32::NAN, 1.2)).unwrap();
        assert!(r.is_nan());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_p_sin() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();

        let shape = F::new(&ctx, s).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();

        for x in [0.0, 1.0, 2.0] {
            let (r, trace) = eval.eval(&tape, &[x]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, &[x]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, &[x]).unwrap();
            assert_eq!(r, x.sin());
            assert!(trace.is_none());
        }

        let y = ctx.y();
        let s = ctx.add(s, y).unwrap();
        let shape = F::new(&ctx, s).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);

        for (x, y) in [(0.0, 1.0), (1.0, 3.0), (2.0, 8.0)] {
            let (r, trace) = eval.eval(&tape, &vs(x, y)).unwrap();
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
        let shape = F::new(&ctx, min).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &vs(1.0, 2.0)).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, &vs(1.0, 3.0)).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, &vs(3.0, 3.5)).unwrap().0, 3.5);
    }

    pub fn test_push() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = F::new(&ctx, min).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &vs(1.0, 2.0)).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, &vs(3.0, 2.0)).unwrap().0, 2.0);

        let next = shape
            .simplify(
                &vec![Choice::Left].into(),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let tape = next.point_tape(Default::default());
        let vs = bind_xy(&tape);
        assert_eq!(eval.eval(&tape, &vs(1.0, 2.0)).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, &vs(3.0, 2.0)).unwrap().0, 3.0);

        let next = shape
            .simplify(
                &vec![Choice::Right].into(),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let tape = next.point_tape(Default::default());
        let vs = bind_xy(&tape);
        assert_eq!(eval.eval(&tape, &vs(1.0, 2.0)).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, &vs(3.0, 2.0)).unwrap().0, 2.0);

        let min = ctx.min(x, 1.0).unwrap();
        let shape = F::new(&ctx, min).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &[0.5]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&tape, &[3.0]).unwrap().0, 1.0);

        let next = shape
            .simplify(
                &vec![Choice::Left].into(),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let tape = next.point_tape(Default::default());
        assert_eq!(eval.eval(&tape, &[0.5]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&tape, &[3.0]).unwrap().0, 3.0);

        let next = shape
            .simplify(
                &vec![Choice::Right].into(),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let tape = next.point_tape(Default::default());
        assert_eq!(eval.eval(&tape, &[0.5]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, &[3.0]).unwrap().0, 1.0);
    }

    pub fn test_basic() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape = F::new(&ctx, x).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &[1.0]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&tape, &[3.0]).unwrap().0, 3.0);

        let shape = F::new(&ctx, y).unwrap();
        let tape = shape.point_tape(Default::default());
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &[2.0]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&tape, &[4.0]).unwrap().0, 4.0);

        let y2 = ctx.mul(y, 2.5).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let shape = F::new(&ctx, sum).unwrap();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_point_eval();
        assert_eq!(eval.eval(&tape, &vs(1.0, 2.0)).unwrap().0, 6.0);
    }

    pub fn test_p_shape_var() {
        let v = Var::new();
        let mut ctx = Context::new();

        let x = ctx.x();
        let y = ctx.y();

        let a = ctx.add(x, y).unwrap();
        let a = ctx.add(a, v).unwrap();

        let s = Shape::<F>::new(&ctx, a).unwrap();

        let mut eval = Shape::<F>::new_point_eval();
        let tape = s.ez_point_tape();
        assert!(eval.eval(&tape, 1.0, 2.0, 0.0).is_err());

        let mut h = HashMap::new();
        assert!(eval.eval_v(&tape, 1.0, 2.0, 0.0, &h).is_err());

        let index = v.index().unwrap();
        h.insert(index, 3.0);
        assert_eq!(eval.eval_v(&tape, 1.0, 2.0, 0.0, &h).unwrap().0, 6.0);
        h.insert(index, 4.0);
        assert_eq!(eval.eval_v(&tape, 1.0, 2.0, 0.0, &h).unwrap().0, 7.0);
    }

    pub fn test_p_stress_n(depth: usize) {
        let (ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD register
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x: Vec<_> = args.clone();
        let y: Vec<_> = x[1..].iter().chain(&x[0..1]).cloned().collect();
        let z: Vec<_> = x[2..].iter().chain(&x[0..2]).cloned().collect();

        let shape = F::new(&ctx, node).unwrap();
        let mut eval = F::new_point_eval();
        let tape = shape.point_tape(Default::default());
        let vs = bind_xyz(&tape);

        let mut out = vec![];
        for i in 0..args.len() {
            out.push(eval.eval(&tape, &vs(x[i], y[i], z[i])).unwrap().0);
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
            cmp.push(eval.eval(&tape, x[i], y[i], z[i]).unwrap().0);
        }

        for (a, b) in out.iter().zip(cmp.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    pub fn test_p_stress() {
        for n in [4, 8, 12, 16, 32] {
            Self::test_p_stress_n(n);
        }
    }

    pub fn test_unary<C: CanonicalUnaryOp>() {
        // Pick a bunch of arguments, some of which are spicy
        let args = test_args();

        let mut ctx = Context::new();
        for v in [ctx.x(), ctx.y(), ctx.z(), ctx.var(Var::new())].into_iter() {
            let node = C::build(&mut ctx, v);

            let shape = F::new(&ctx, node).unwrap();
            let mut eval = F::new_point_eval();
            let tape = shape.point_tape(Default::default());

            for &a in args.iter() {
                let (o, trace) = eval.eval(&tape, &[a]).unwrap();
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

    fn compare_point_results<C: CanonicalBinaryOp>(
        lhs: f32,
        rhs: f32,
        out: f32,
        g: impl Fn(f32, f32) -> f32,
        name: &str,
    ) {
        let value = g(lhs, rhs);
        let err = (value - out).abs();
        assert!(
            (out == value)
                || C::discontinuous_at(lhs, rhs)
                || err < 1e-6
                || value.is_nan() && out.is_nan(),
            "mismatch in '{name}' at ({lhs}, {rhs}): \
                            {value} != {out} ({err})"
        )
    }

    pub fn test_binary_reg_reg<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let va = Var::new();
        let vb = Var::new();

        let name = format!("{}(reg, reg)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                let node = C::build(&mut ctx, va, vb);

                let shape = F::new(&ctx, node).unwrap();
                let mut eval = F::new_point_eval();
                let tape = shape.point_tape(Default::default());
                let vars = tape.vars();

                let i_index = vars[&va];
                let j_index = vars[&vb];
                assert_ne!(i_index, j_index);

                let mut args = [0.0; 2];
                args[j_index] = rhs;
                args[i_index] = lhs;

                let (out, _trace) = eval.eval(&tape, &args).unwrap();

                Self::compare_point_results::<C>(
                    lhs,
                    rhs,
                    out,
                    C::eval_reg_reg_f32,
                    &name,
                );
            }
        }

        for &lhs in args.iter() {
            let node = C::build(&mut ctx, va, va);

            let shape = F::new(&ctx, node).unwrap();
            let mut eval = F::new_point_eval();
            let tape = shape.point_tape(Default::default());
            let vars = tape.vars();

            let i_index = vars[&va];
            assert_eq!(i_index, 0);

            let args = [lhs];
            let (out, _trace) = eval.eval(&tape, &args).unwrap();

            Self::compare_point_results::<C>(
                lhs,
                lhs,
                out,
                C::eval_reg_reg_f32,
                &name,
            );
        }
    }

    pub fn test_binary_reg_imm<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let va = Var::new();

        let name = format!("{}(reg, imm)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                let node = C::build(&mut ctx, va, rhs);

                let shape = F::new(&ctx, node).unwrap();
                let mut eval = F::new_point_eval();
                let tape = shape.point_tape(Default::default());

                let (out, _trace) = eval.eval(&tape, &[lhs]).unwrap();

                Self::compare_point_results::<C>(
                    lhs,
                    rhs,
                    out,
                    C::eval_reg_imm_f32,
                    &name,
                );
            }
        }
    }

    pub fn test_binary_imm_reg<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let va = Var::new();

        let name = format!("{}(imm, reg)", C::NAME);
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                let node = C::build(&mut ctx, lhs, va);

                let shape = F::new(&ctx, node).unwrap();
                let mut eval = F::new_point_eval();
                let tape = shape.point_tape(Default::default());

                let (out, _trace) = eval.eval(&tape, &[rhs]).unwrap();

                Self::compare_point_results::<C>(
                    lhs,
                    rhs,
                    out,
                    C::eval_imm_reg_f32,
                    &name,
                );
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
        $crate::point_test!(test_p_and, $t);
        $crate::point_test!(test_p_or, $t);
        $crate::point_test!(basic_interpreter, $t);
        $crate::point_test!(test_push, $t);
        $crate::point_test!(test_basic, $t);
        $crate::point_test!(test_p_shape_var, $t);
        $crate::point_test!(test_p_stress, $t);

        mod p_unary {
            use super::*;
            $crate::all_unary_tests!(
                $crate::eval::test::point::TestPoint::<$t>
            );
        }

        mod p_binary {
            use super::*;
            $crate::all_binary_tests!(
                $crate::eval::test::point::TestPoint::<$t>
            );
        }
    };
}
