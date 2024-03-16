//! Test suite for partial derivative (gradient) evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
use super::{build_stress_fn, test_args};
use crate::{
    context::{Context, Node},
    eval::{
        types::Grad, BulkEvaluator, EzShape, MathShape, Shape, ShapeVars, Vars,
    },
    Error,
};

macro_rules! grad_slice_unary {
    (Context::$i:ident, $t:expr) => {
        Self::test_unary(Context::$i, $t, stringify!($i));
    };
}

macro_rules! grad_slice_binary {
    (Context::$i:ident, $t:expr) => {
        Self::test_binary_reg_reg(Context::$i, $t, stringify!($i));
        Self::test_binary_reg_imm(Context::$i, $t, stringify!($i));
        Self::test_binary_imm_reg(Context::$i, $t, stringify!($i));
    };
}

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
        name: &'static str,
    ) {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        for (i, v) in [ctx.x(), ctx.y(), ctx.z()].into_iter().enumerate() {
            let node = f(&mut ctx, v).unwrap();

            let shape = S::new(&ctx, node).unwrap();
            let mut eval = S::new_grad_slice_eval();
            let tape = shape.ez_grad_slice_tape();

            let out = match i {
                0 => eval.eval(&tape, &args, &zero, &zero, &[]),
                1 => eval.eval(&tape, &zero, &args, &zero, &[]),
                2 => eval.eval(&tape, &zero, &zero, &args, &[]),
                _ => unreachable!(),
            }
            .unwrap();
            for (a, &o) in args.iter().zip(out.iter()) {
                let v = g(*a as f64);
                let err = (v as f32 - o.v).abs();
                let err_frac = err / (v.abs() as f32).max(o.v.abs());
                assert!(
                    (o.v == v as f32)
                        || err < 1e-6
                        || err_frac < 1e-6
                        || (v.is_nan() && o.v.is_nan()),
                    "mismatch in '{name}' at {a}: {v} != {o} ({err})"
                );

                let grad = o.d(i);
                if !v.is_nan() && grad < 1e9 && !grad.is_infinite() {
                    let d = g(*a as f64 + 1e-8);
                    let estimated_gradient = (d - v) / 1e-8;
                    let err = (estimated_gradient as f32 - grad).abs();
                    assert!(
                        err < 1e-3,
                        "gradient estimate mismatch in '{name}' at {a}:
                        {estimated_gradient} != {grad} ({err})"
                    );
                }
            }
        }
    }

    pub fn compare_grad_results(
        i: usize,
        j: usize,
        lhs: &[f32],
        rhs: &[f32],
        out: &[Grad],
        g: impl Fn(f64, f64) -> f64,
        name: &str,
    ) {
        for ((a, b), &o) in lhs.iter().zip(rhs).zip(out.iter()) {
            let v = g(*a as f64, *b as f64);
            let err = (v as f32 - o.v).abs();
            let err_frac = err / (v.abs() as f32).max(o.v.abs());
            assert!(
                (o.v == v as f32)
                    || err < 1e-6
                    || err_frac < 1e-6
                    || (v.is_nan() && o.v.is_nan()),
                "value mismatch in '{name}' at ({a}, {b}): {v} != {o} ({err})"
            );

            if v.is_nan() {
                continue;
            }

            const EPSILON: f64 = 1e-8;
            let a = *a as f64;
            let b = *b as f64;
            if i == j {
                let grad = o.d(i);
                if grad < 1e9 && !grad.is_infinite() {
                    let d = g(a + EPSILON, b + EPSILON);
                    let est_grad = (d - v) / EPSILON;
                    let mut err = (est_grad as f32 - grad).abs();

                    let d = g(a - EPSILON, b);
                    let est_grad = (v - d) / EPSILON;
                    err = err.min((est_grad as f32 - grad).abs());
                    assert!(
                        err < 1e-3,
                        "gradient estimate mismatch in '{name}' at \
                         ({a} + epsilon, {b}): \
                         {est_grad} != {grad} ({err})"
                    );
                }
            } else {
                if i < 3 {
                    let grad = o.d(i);
                    if grad < 1e9 && !grad.is_infinite() {
                        // Check both +epsilon and -epsilon positions, because
                        // they may be different if there are C1 discontinuities
                        let d = g(a + EPSILON, b);
                        let est_grad = (d - v) / EPSILON;
                        let mut err = (est_grad as f32 - grad).abs();

                        let d = g(a - EPSILON, b);
                        let est_grad = (v - d) / EPSILON;
                        err = err.min((est_grad as f32 - grad).abs());
                        assert!(
                            err < 1e-3,
                            "gradient estimate mismatch in '{name}' at \
                             ({a} + epsilon, {b}): \
                             {est_grad} != {grad} ({err})"
                        );
                    }
                }
                if j < 3 {
                    let grad = o.d(j);
                    if grad < 1e9 && !grad.is_infinite() {
                        let d = g(a, b + EPSILON);
                        let est_grad = (d - v) / EPSILON;
                        let mut err = (est_grad as f32 - grad).abs();

                        let d = g(a, b - EPSILON);
                        let est_grad = (v - d) / EPSILON;
                        err = err.min((est_grad as f32 - grad).abs());
                        assert!(
                            err < 1e-3,
                            "gradient estimate mismatch in '{name}' at \
                             ({a}, {b} + epsilon): \
                             {est_grad} != {grad} ({err})"
                        );
                    }
                }
            }
        }
    }

    pub fn test_binary_reg_reg(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f64, f64) -> f64,
        name: &'static str,
    ) {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];
        let name = format!("{name}(reg, reg)");
        for rot in 0..args.len() {
            let mut rgsa = args.clone();
            rgsa.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for (j, &u) in inputs.iter().enumerate() {
                    let node = f(&mut ctx, v, u).unwrap();

                    let shape = S::new(&ctx, node).unwrap();
                    let mut eval = S::new_grad_slice_eval();
                    let tape = shape.ez_grad_slice_tape();

                    let out = match (i, j) {
                        (0, 0) => eval.eval(&tape, &args, &zero, &zero, &[]),
                        (0, 1) => eval.eval(&tape, &args, &rgsa, &zero, &[]),
                        (0, 2) => eval.eval(&tape, &args, &zero, &rgsa, &[]),
                        (1, 0) => eval.eval(&tape, &rgsa, &args, &zero, &[]),
                        (1, 1) => eval.eval(&tape, &zero, &args, &zero, &[]),
                        (1, 2) => eval.eval(&tape, &zero, &args, &rgsa, &[]),
                        (2, 0) => eval.eval(&tape, &rgsa, &zero, &args, &[]),
                        (2, 1) => eval.eval(&tape, &zero, &rgsa, &args, &[]),
                        (2, 2) => eval.eval(&tape, &zero, &zero, &args, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let rhs = if i == j { &args } else { &rgsa };
                    Self::compare_grad_results(
                        i, j, &args, rhs, out, &g, &name,
                    );
                }
            }
        }
    }

    fn test_binary_reg_imm(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f64, f64) -> f64,
        name: &'static str,
    ) {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{name}(reg, imm)");
        for rot in 0..args.len() {
            let mut args = args.clone();
            args.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for rhs in args.iter() {
                    let c = ctx.constant(*rhs as f64);
                    let node = f(&mut ctx, v, c).unwrap();

                    let shape = S::new(&ctx, node).unwrap();
                    let mut eval = S::new_grad_slice_eval();
                    let tape = shape.ez_grad_slice_tape();

                    let out = match i {
                        0 => eval.eval(&tape, &args, &zero, &zero, &[]),
                        1 => eval.eval(&tape, &zero, &args, &zero, &[]),
                        2 => eval.eval(&tape, &zero, &zero, &args, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let rhs = vec![*rhs; out.len()];
                    Self::compare_grad_results(
                        i, 3, &args, &rhs, out, &g, &name,
                    );
                }
            }
        }
    }

    fn test_binary_imm_reg(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f64, f64) -> f64,
        name: &'static str,
    ) {
        let args = test_args();
        let zero = vec![0.0; args.len()];

        let mut ctx = Context::new();
        let inputs = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{name}(reg, imm)");
        for rot in 0..args.len() {
            let mut args = args.clone();
            args.rotate_left(rot);
            for (i, &v) in inputs.iter().enumerate() {
                for lhs in args.iter() {
                    let c = ctx.constant(*lhs as f64);
                    let node = f(&mut ctx, c, v).unwrap();

                    let shape = S::new(&ctx, node).unwrap();
                    let mut eval = S::new_grad_slice_eval();
                    let tape = shape.ez_grad_slice_tape();

                    let out = match i {
                        0 => eval.eval(&tape, &args, &zero, &zero, &[]),
                        1 => eval.eval(&tape, &zero, &args, &zero, &[]),
                        2 => eval.eval(&tape, &zero, &zero, &args, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();

                    let lhs = vec![*lhs; out.len()];
                    Self::compare_grad_results(
                        3, i, &lhs, &args, out, &g, &name,
                    );
                }
            }
        }
    }

    pub fn test_g_unary_ops() {
        grad_slice_unary!(Context::neg, |v| -v);
        grad_slice_unary!(Context::recip, |v| 1.0 / v);
        grad_slice_unary!(Context::abs, |v| v.abs());
        grad_slice_unary!(Context::sin, |v| v.sin());
        grad_slice_unary!(Context::cos, |v| v.cos());
        grad_slice_unary!(Context::tan, |v| v.tan());
        grad_slice_unary!(Context::asin, |v| v.asin());
        grad_slice_unary!(Context::acos, |v| v.acos());
        grad_slice_unary!(Context::atan, |v| v.atan());
        grad_slice_unary!(Context::exp, |v| v.exp());
        grad_slice_unary!(Context::ln, |v| v.ln());
        grad_slice_unary!(Context::square, |v| v * v);
        grad_slice_unary!(Context::sqrt, |v| v.sqrt());
    }

    pub fn test_g_binary_ops() {
        grad_slice_binary!(Context::add, |a, b| a + b);
        grad_slice_binary!(Context::sub, |a, b| a - b);

        // Multiplication short-circuits to 0, which means that
        // 0 (constant) * NaN = 0
        Self::test_binary_reg_reg(Context::mul, |a, b| a * b, "mul");
        Self::test_binary_reg_imm(
            Context::mul,
            |a, b| if b == 0.0 { b } else { a * b },
            "mul",
        );
        Self::test_binary_imm_reg(
            Context::mul,
            |a, b| if a == 0.0 { a } else { a * b },
            "mul",
        );

        // Multiplication short-circuits to 0, which means that
        // 0 (constant) / NaN = 0
        Self::test_binary_reg_reg(Context::div, |a, b| a / b, "div");
        Self::test_binary_reg_imm(Context::div, |a, b| a / b, "div");
        Self::test_binary_imm_reg(
            Context::div,
            |a, b| if a == 0.0 { a } else { a / b },
            "div",
        );

        grad_slice_binary!(Context::min, |a, b| if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        });
        grad_slice_binary!(Context::max, |a, b| if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        });
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
        $crate::grad_test!(test_g_binary_ops, $t);
    };
}
