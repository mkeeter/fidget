//! Test suite for partial derivative (gradient) evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
use super::build_stress_fn;
use crate::{
    context::{Context, Node},
    eval::{
        types::Grad, BulkEvaluator, EzShape, MathShape, Shape, ShapeVars, Vars,
    },
    Error,
};

/// Helper struct to put constrains on our `Shape` object
pub struct TestGradSlice<S>(std::marker::PhantomData<*const S>);

impl<S> TestGradSlice<S>
where
    S: Shape + MathShape + ShapeVars,
{
    pub fn test_g_x() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = S::new(&ctx, x).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_y() {
        let mut ctx = Context::new();
        let y = ctx.y();
        let shape = S::new(&ctx, y).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_z() {
        let mut ctx = Context::new();
        let z = ctx.z();
        let shape = S::new(&ctx, z).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(4.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.square(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[0.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 4.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[3.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(9.0, 6.0, 0.0, 0.0)
        );
    }

    pub fn test_g_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.abs(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[-2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, -1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sqrt(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 0.5, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_sin() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_grad_slice_tape();

        let mut eval = S::new_grad_slice_eval();
        let v = eval
            .eval(&tape, &[1.0, 2.0, 3.0], &[0.0; 3], &[0.0; 3], &[])
            .unwrap();
        v[0].compare_eq(Grad::new(1f32.sin(), 1f32.cos(), 0.0, 0.0));
        v[1].compare_eq(Grad::new(2f32.sin(), 2f32.cos(), 0.0, 0.0));
        v[2].compare_eq(Grad::new(3f32.sin(), 3f32.cos(), 0.0, 0.0));

        let y = ctx.y();
        let y = ctx.mul(y, 2.0).unwrap();
        let s = ctx.sin(y).unwrap();
        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_grad_slice_tape();
        let v = eval
            .eval(&tape, &[0.0; 3], &[1.0, 2.0, 3.0], &[0.0; 3], &[])
            .unwrap();
        v[0].compare_eq(Grad::new(2f32.sin(), 0.0, 2.0 * 2f32.cos(), 0.0));
        v[1].compare_eq(Grad::new(4f32.sin(), 0.0, 2.0 * 4f32.cos(), 0.0));
        v[2].compare_eq(Grad::new(6f32.sin(), 0.0, 2.0 * 6f32.cos(), 0.0));
    }

    pub fn test_g_mul() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let s = ctx.mul(x, y).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 4.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(8.0, 2.0, 4.0, 0.0)
        );
    }

    pub fn test_g_div() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.div(x, 2.0).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.5, 0.0, 0.0)
        );
    }

    pub fn test_g_recip() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.recip(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, -1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, -0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.min(x, y).unwrap();
        let shape = S::new(&ctx, m).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_min_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();
        let min = ctx.min(x, y).unwrap();
        let max = ctx.max(min, z).unwrap();
        let shape = S::new(&ctx, max).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[3.0], &[5.0], &[]).unwrap()[0],
            Grad::new(5.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.max(x, y).unwrap();
        let shape = S::new(&ctx, m).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_circle() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let sqrt = ctx.sqrt(sum).unwrap();
        let sub = ctx.sub(sqrt, 0.5).unwrap();
        let shape = S::new(&ctx, sub).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&tape, &[0.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_var() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let shape = S::new(&ctx, a).unwrap();
        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&tape, &[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            2.0.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let div = ctx.div(sum, 2.0).unwrap();
        let shape = S::new(&ctx, div).unwrap();

        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();
        assert_eq!(
            eval.eval(&tape, &[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&tape, &[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            1.5.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let mut vars = Vars::new(shape.vars());
        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();

        assert_eq!(
            eval.eval(
                &tape,
                &[0.0],
                &[0.0],
                &[0.0],
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap()[0],
            2.0.into()
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
            2.0.into()
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
            0.5.into(),
        );
    }

    pub fn test_g_stress_n(depth: usize) {
        let (ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD registers
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x = args.clone();
        let y: Vec<f32> =
            args[1..].iter().chain(&args[0..1]).cloned().collect();
        let z: Vec<f32> =
            args[2..].iter().chain(&args[0..2]).cloned().collect();

        let shape = S::new(&ctx, node).unwrap();
        let mut eval = S::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();

        let out = eval.eval(&tape, &x, &y, &z, &[]).unwrap();

        // Compare values (the `.v` term) with the context's evaluator
        for (i, v) in out.iter().cloned().enumerate() {
            let q = ctx
                .eval_xyz(node, x[i] as f64, y[i] as f64, z[i] as f64)
                .unwrap();
            let err = (v.v as f64 - q).abs();
            assert!(
                err < 1e-2, // generous error bounds, for the 512-op case
                "mismatch at index {i} ({}, {}, {}): {v:?} != {q} [{err}], {}",
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
        let mut eval = VmShape::new_grad_slice_eval();
        let tape = shape.ez_grad_slice_tape();

        let cmp = eval.eval(&tape, &x, &y, &z, &[]).unwrap();
        for (a, b) in out.iter().zip(cmp.iter()) {
            a.compare_eq(*b)
        }
    }

    pub fn test_g_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32] {
            Self::test_g_stress_n(n);
        }
    }

    pub fn test_unary(
        f: impl Fn(&mut Context, Node) -> Result<Node, Error>,
        g: impl Fn(f64) -> f64,
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
            let mut eval = S::new_grad_slice_eval();
            let tape = shape.ez_grad_slice_tape();

            let out = eval.eval(&tape, &args, &args, &args, &[]).unwrap();
            for (a, &o) in args.iter().zip(out.iter()) {
                let v = g(*a as f64);
                let err = (v as f32 - o.v).abs();
                assert!(
                    (o.v == v as f32)
                        || err < 1e-6
                        || (v.is_nan() && o.v.is_nan()),
                    "mismatch at {a}: {v} != {o} ({err})"
                );

                let grad = o.d(i);
                if !v.is_nan() && grad < 1e9 && !grad.is_infinite() {
                    let d = g(*a as f64 + 1e-8);
                    let estimated_gradient = (d - v) / 1e-8;
                    let err = (estimated_gradient as f32 - grad).abs();
                    assert!(
                        err < 1e-3,
                        "gradient estimate mismatch at {a}:
                        {estimated_gradient} != {grad} ({err})"
                    );
                }
            }
        }
    }

    pub fn test_g_unary_ops() {
        Self::test_unary(Context::sin, |v| v.sin());
        Self::test_unary(Context::cos, |v| v.cos());
        Self::test_unary(Context::tan, |v| v.tan());
        Self::test_unary(Context::asin, |v| v.asin());
        Self::test_unary(Context::acos, |v| v.acos());
        Self::test_unary(Context::atan, |v| v.atan());
        Self::test_unary(Context::exp, |v| v.exp());
        Self::test_unary(Context::ln, |v| v.ln());
        Self::test_unary(Context::square, |v| v * v);
        Self::test_unary(Context::sqrt, |v| v.sqrt());
    }
}

#[macro_export]
macro_rules! grad_test {
    ($i:ident, $t:ty) => {
        #[test]
        fn $i() {
            $crate::eval::test::grad_slice::TestGradSlice::<$t>::$i()
        }
    };
}

#[macro_export]
macro_rules! grad_slice_tests {
    ($t:ty) => {
        $crate::grad_test!(test_g_circle, $t);
        $crate::grad_test!(test_g_x, $t);
        $crate::grad_test!(test_g_y, $t);
        $crate::grad_test!(test_g_z, $t);
        $crate::grad_test!(test_g_abs, $t);
        $crate::grad_test!(test_g_square, $t);
        $crate::grad_test!(test_g_sqrt, $t);
        $crate::grad_test!(test_g_sin, $t);
        $crate::grad_test!(test_g_mul, $t);
        $crate::grad_test!(test_g_min, $t);
        $crate::grad_test!(test_g_max, $t);
        $crate::grad_test!(test_g_min_max, $t);
        $crate::grad_test!(test_g_div, $t);
        $crate::grad_test!(test_g_recip, $t);
        $crate::grad_test!(test_g_var, $t);
        $crate::grad_test!(test_g_stress, $t);
        $crate::grad_test!(test_g_unary_ops, $t);
    };
}
