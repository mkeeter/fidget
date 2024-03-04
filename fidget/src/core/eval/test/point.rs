//! Test suite for single-point tracing evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for point evaluators; otherwise, the module has no public exports.
use super::build_stress_fn;
use crate::{
    context::{Context, Node},
    eval::{EzShape, MathShape, Shape, ShapeVars, TracingEvaluator, Vars},
    vm::Choice,
    Error,
};

macro_rules! point_unary {
    (Context::$i:ident, $t:expr) => {
        Self::test_unary(Context::$i, $t, stringify!($i));
    };
}

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

    pub fn test_unary(
        f: impl Fn(&mut Context, Node) -> Result<Node, Error>,
        g: impl Fn(f32) -> f32,
        name: &'static str,
    ) {
        // Pick a bunch of arguments, some of which are spicy
        let mut args =
            (-32..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        args.push(0.0);
        args.push(1.0);
        args.push(std::f32::consts::PI);
        args.push(std::f32::consts::FRAC_PI_2);
        args.push(std::f32::consts::FRAC_1_PI);
        args.push(std::f32::consts::SQRT_2);
        args.push(f32::NAN);

        let mut ctx = Context::new();
        for (i, v) in [ctx.x(), ctx.y(), ctx.z()].into_iter().enumerate() {
            let node = f(&mut ctx, v).unwrap();

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
                let v = g(a);
                let err = (v - o).abs();
                assert!(
                    (o == v) || err < 1e-6 || (v.is_nan() && o.is_nan()),
                    "mismatch in '{name}' at {a}: {v} != {o} ({err})"
                )
            }
        }
    }

    pub fn test_p_unary_ops() {
        point_unary!(Context::sin, |v| v.sin());
        point_unary!(Context::cos, |v| v.cos());
        point_unary!(Context::tan, |v| v.tan());
        point_unary!(Context::asin, |v| v.asin());
        point_unary!(Context::acos, |v| v.acos());
        point_unary!(Context::atan, |v| v.atan());
        point_unary!(Context::exp, |v| v.exp());
        point_unary!(Context::ln, |v| v.ln());
        point_unary!(Context::square, |v| v * v);
        point_unary!(Context::sqrt, |v| v.sqrt());
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
    };
}
