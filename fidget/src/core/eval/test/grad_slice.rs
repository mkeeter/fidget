//! Test suite for partial derivative (gradient) evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
use super::{build_stress_fn, test_args, CanonicalBinaryOp, CanonicalUnaryOp};
use crate::{
    context::Context,
    eval::{BulkEvaluator, Function, MathFunction, Tape},
    types::Grad,
    var::Var,
    vm::VmFunction,
};

/// Epsilon for gradient estimates
const EPSILON: f64 = 1e-8;

/// Helper struct to put constrains on our `Shape` object
pub struct TestGradSlice<F>(std::marker::PhantomData<*const F>);

impl<F: Function + MathFunction> TestGradSlice<F> {
    fn eval_xyz(
        tape: &<<F as Function>::GradSliceEval as BulkEvaluator>::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
    ) -> Vec<Grad> {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        let xs: Vec<_> =
            xs.iter().map(|x| Grad::new(*x, 1.0, 0.0, 0.0)).collect();
        let ys: Vec<_> =
            ys.iter().map(|y| Grad::new(*y, 0.0, 1.0, 0.0)).collect();
        let zs: Vec<_> =
            zs.iter().map(|z| Grad::new(*z, 0.0, 0.0, 1.0)).collect();

        let vars = tape.vars();
        let mut out = [
            vec![Grad::from(0.0); xs.len()],
            vec![Grad::from(0.0); ys.len()],
            vec![Grad::from(0.0); zs.len()],
        ];
        if let Some(ix) = vars.get(&Var::X) {
            out[ix] = xs;
        }
        if let Some(iy) = vars.get(&Var::Y) {
            out[iy] = ys;
        }
        if let Some(iz) = vars.get(&Var::Z) {
            out[iz] = zs;
        }

        let mut eval = F::new_grad_slice_eval();
        eval.eval(tape, &out).unwrap().to_owned()
    }

    pub fn test_g_x() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let shape = F::new(&ctx, x).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[4.0])[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_y() {
        let mut ctx = Context::new();
        let y = ctx.y();
        let shape = F::new(&ctx, y).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[4.0])[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_z() {
        let mut ctx = Context::new();
        let z = ctx.z();
        let shape = F::new(&ctx, z).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[4.0])[0],
            Grad::new(4.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.square(x).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[0.0], &[0.0], &[0.0])[0],
            Grad::new(0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(1.0, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[0.0], &[0.0])[0],
            Grad::new(4.0, 4.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[3.0], &[0.0], &[0.0])[0],
            Grad::new(9.0, 6.0, 0.0, 0.0)
        );
    }

    pub fn test_g_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.abs(x).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[0.0], &[0.0])[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[-2.0], &[0.0], &[0.0])[0],
            Grad::new(2.0, -1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sqrt(x).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(1.0, 0.5, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[0.0], &[0.0])[0],
            Grad::new(2.0, 0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_sin() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();
        let shape = F::new(&ctx, s).unwrap();
        let tape = shape.grad_slice_tape(Default::default());

        let v = Self::eval_xyz(&tape, &[1.0, 2.0, 3.0], &[0.0; 3], &[0.0; 3]);
        v[0].compare_eq(Grad::new(1f32.sin(), 1f32.cos(), 0.0, 0.0));
        v[1].compare_eq(Grad::new(2f32.sin(), 2f32.cos(), 0.0, 0.0));
        v[2].compare_eq(Grad::new(3f32.sin(), 3f32.cos(), 0.0, 0.0));

        let y = ctx.y();
        let y = ctx.mul(y, 2.0).unwrap();
        let s = ctx.sin(y).unwrap();
        let shape = F::new(&ctx, s).unwrap();
        let tape = shape.grad_slice_tape(Default::default());
        let v = Self::eval_xyz(&tape, &[0.0; 3], &[1.0, 2.0, 3.0], &[0.0; 3]);
        v[0].compare_eq(Grad::new(2f32.sin(), 0.0, 2.0 * 2f32.cos(), 0.0));
        v[1].compare_eq(Grad::new(4f32.sin(), 0.0, 2.0 * 4f32.cos(), 0.0));
        v[2].compare_eq(Grad::new(6f32.sin(), 0.0, 2.0 * 6f32.cos(), 0.0));
    }

    pub fn test_g_mul() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let s = ctx.mul(x, y).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(0.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[0.0], &[1.0], &[0.0])[0],
            Grad::new(0.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[1.0], &[0.0])[0],
            Grad::new(4.0, 1.0, 4.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[2.0], &[0.0])[0],
            Grad::new(8.0, 2.0, 4.0, 0.0)
        );
    }

    pub fn test_g_div() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.div(x, 2.0).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(0.5, 0.5, 0.0, 0.0)
        );
    }

    pub fn test_g_recip() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.recip(x).unwrap();
        let shape = F::new(&ctx, s).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(1.0, -1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[0.0], &[0.0])[0],
            Grad::new(0.5, -0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.min(x, y).unwrap();
        let shape = F::new(&ctx, m).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[0.0])[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[3.0], &[0.0])[0],
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
        let shape = F::new(&ctx, max).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[0.0])[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[3.0], &[0.0])[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[3.0], &[5.0])[0],
            Grad::new(5.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.max(x, y).unwrap();
        let shape = F::new(&ctx, m).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[3.0], &[0.0])[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[4.0], &[3.0], &[0.0])[0],
            Grad::new(4.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_not() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let m = ctx.not(x).unwrap();
        let shape = F::new(&ctx, m).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[0.0], &[0.0], &[0.0])[0],
            Grad::new(1.0, 0.0, 0.0, 0.0)
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
        let shape = F::new(&ctx, sub).unwrap();

        let tape = shape.grad_slice_tape(Default::default());
        assert_eq!(
            Self::eval_xyz(&tape, &[1.0], &[0.0], &[0.0])[0],
            Grad::new(0.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[0.0], &[1.0], &[0.0])[0],
            Grad::new(0.5, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[2.0], &[0.0], &[0.0])[0],
            Grad::new(1.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            Self::eval_xyz(&tape, &[0.0], &[2.0], &[0.0])[0],
            Grad::new(1.5, 0.0, 1.0, 0.0)
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

        let shape = F::new(&ctx, node).unwrap();
        let tape = shape.grad_slice_tape(Default::default());

        let out = Self::eval_xyz(&tape, &x, &y, &z);

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
        let shape = VmFunction::new(&ctx, node).unwrap();
        let tape = shape.grad_slice_tape(Default::default());

        let cmp = TestGradSlice::<VmFunction>::eval_xyz(&tape, &x, &y, &z);
        for (a, b) in out.iter().zip(cmp.iter()) {
            a.compare_eq(*b)
        }
    }

    pub fn test_g_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32] {
            Self::test_g_stress_n(n);
        }
    }

    pub fn test_unary<C: CanonicalUnaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let v = ctx.var(Var::new());
        let node = C::build(&mut ctx, v);
        let shape = F::new(&ctx, node).unwrap();
        let tape = shape.grad_slice_tape(Default::default());
        let mut eval = F::new_grad_slice_eval();

        for i in 0..3 {
            let args = args
                .iter()
                .map(|&v| match i {
                    0 => Grad::new(v, 1.0, 0.0, 0.0),
                    1 => Grad::new(v, 0.0, 1.0, 0.0),
                    2 => Grad::new(v, 0.0, 0.0, 1.0),
                    _ => unreachable!(),
                })
                .collect::<Vec<Grad>>();
            let out = eval.eval(&tape, &[args.as_slice()]).unwrap();
            for (a, &o) in args.iter().zip(out.iter()) {
                let v = C::eval_f64(a.v as f64);
                let err = (v as f32 - o.v).abs();
                let err_frac = err / (v.abs() as f32).max(o.v.abs());
                assert!(
                    o.v == v as f32
                        || err < 1e-6
                        || err_frac < 1e-6
                        || (v.is_nan() && o.v.is_nan()),
                    "mismatch in '{}' at {a}: {v} != {o} ({err})",
                    C::NAME,
                );

                if C::discontinuous_at(a.v) {
                    continue;
                }

                let grad = o.d(i);
                if !v.is_nan() && grad < 1e9 && !grad.is_infinite() {
                    let a = a.v as f64;
                    let d = C::eval_f64(a + EPSILON);
                    let est_grad = (d - v) / EPSILON;
                    let mut err = (est_grad as f32 - grad).abs();

                    let d = C::eval_f64(a - EPSILON);
                    let est_grad = (v - d) / EPSILON;
                    err = err.min((est_grad as f32 - grad).abs());
                    assert!(
                        err.min(err / grad.abs()) < 1e-3,
                        "gradient estimate mismatch in '{}' at {a} => {o:?}:
                        {est_grad} != {grad} ({err})",
                        C::NAME,
                    );
                }
            }
        }
    }

    pub fn compare_grad_results<C: CanonicalBinaryOp>(
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
                "value mismatch in '{name}' at ({a}, {b}) => {o:?}: \
                 {v} != {o} ({err})"
            );

            if v.is_nan() {
                continue;
            }

            if C::discontinuous_at(*a, *b) {
                continue;
            }

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
                        err.min(err / grad.abs()) < 1e-3,
                        "gradient estimate mismatch in '{name}' at \
                         ({a} + epsilon, {b}) => {o:?}: \
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
                            err.min(err / grad.abs()) < 1e-3,
                            "gradient estimate mismatch in '{name}' at \
                             ({a} + epsilon, {b}) => {o:?}: \
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
                            err.min(err / grad.abs()) < 1e-3,
                            "gradient estimate mismatch in '{name}' at \
                             ({a}, {b} + epsilon) => {o:?}: \
                             {est_grad} != {grad} ({err})"
                        );
                    }
                }
            }
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

                    let shape = F::new(&ctx, node).unwrap();
                    let tape = shape.grad_slice_tape(Default::default());

                    let out = match (i, j) {
                        (0, 0) => Self::eval_xyz(&tape, &args, &zero, &zero),
                        (0, 1) => Self::eval_xyz(&tape, &args, &rgsa, &zero),
                        (0, 2) => Self::eval_xyz(&tape, &args, &zero, &rgsa),
                        (1, 0) => Self::eval_xyz(&tape, &rgsa, &args, &zero),
                        (1, 1) => Self::eval_xyz(&tape, &zero, &args, &zero),
                        (1, 2) => Self::eval_xyz(&tape, &zero, &args, &rgsa),
                        (2, 0) => Self::eval_xyz(&tape, &rgsa, &zero, &args),
                        (2, 1) => Self::eval_xyz(&tape, &zero, &rgsa, &args),
                        (2, 2) => Self::eval_xyz(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    };

                    let rhs = if i == j { &args } else { &rgsa };
                    Self::compare_grad_results::<C>(
                        i,
                        j,
                        &args,
                        rhs,
                        &out,
                        C::eval_reg_reg_f64,
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
                    let node = C::build(&mut ctx, v, *rhs);

                    let shape = F::new(&ctx, node).unwrap();
                    let tape = shape.grad_slice_tape(Default::default());

                    let out = match i {
                        0 => Self::eval_xyz(&tape, &args, &zero, &zero),
                        1 => Self::eval_xyz(&tape, &zero, &args, &zero),
                        2 => Self::eval_xyz(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    };

                    let rhs = vec![*rhs; out.len()];
                    Self::compare_grad_results::<C>(
                        i,
                        3,
                        &args,
                        &rhs,
                        &out,
                        C::eval_reg_imm_f64,
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
                    let node = C::build(&mut ctx, *lhs, v);

                    let shape = F::new(&ctx, node).unwrap();
                    let tape = shape.grad_slice_tape(Default::default());

                    let out = match i {
                        0 => Self::eval_xyz(&tape, &args, &zero, &zero),
                        1 => Self::eval_xyz(&tape, &zero, &args, &zero),
                        2 => Self::eval_xyz(&tape, &zero, &zero, &args),
                        _ => unreachable!(),
                    };

                    let lhs = vec![*lhs; out.len()];
                    Self::compare_grad_results::<C>(
                        3,
                        i,
                        &lhs,
                        &args,
                        &out,
                        C::eval_imm_reg_f64,
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
        $crate::grad_test!(test_g_not, $t);
        $crate::grad_test!(test_g_div, $t);
        $crate::grad_test!(test_g_recip, $t);
        $crate::grad_test!(test_g_stress, $t);

        mod g_unary {
            use super::*;
            $crate::all_unary_tests!(
                $crate::eval::test::grad_slice::TestGradSlice::<$t>
            );
        }

        mod g_binary {
            use super::*;
            $crate::all_binary_tests!(
                $crate::eval::test::grad_slice::TestGradSlice::<$t>
            );
        }
    };
}
