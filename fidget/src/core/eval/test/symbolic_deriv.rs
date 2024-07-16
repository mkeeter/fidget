use super::{test_args, CanonicalBinaryOp, CanonicalUnaryOp};
use crate::{
    context::Context,
    eval::{BulkEvaluator, Function, MathFunction, Tape},
    types::Grad,
    var::Var,
    vm::VmFunction,
};

/// Helper struct to put constrains on our `Shape` object
pub struct TestSymbolicDerivs;

impl TestSymbolicDerivs {
    pub fn test_unary<C: CanonicalUnaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let v = ctx.var(Var::new());
        let node = C::build(&mut ctx, v);
        let shape = VmFunction::new(&ctx, node).unwrap();
        let tape = shape.grad_slice_tape(Default::default());
        let mut eval = VmFunction::new_grad_slice_eval();

        // Test symbolic differentiation at the same time
        let node_deriv = ctx.deriv(node, ctx.get_var(v).unwrap()).unwrap();
        let shape_deriv = VmFunction::new(&ctx, node_deriv).unwrap();
        let tape_deriv = shape_deriv.float_slice_tape(Default::default());
        let mut eval_deriv = VmFunction::new_float_slice_eval();

        let args_g = args
            .iter()
            .map(|&v| Grad::new(v, 1.0, 0.0, 0.0))
            .collect::<Vec<_>>();
        let out = eval.eval(&tape, &[args_g.as_slice()]).unwrap();

        // Check symbolic differentiation results
        let out_deriv =
            eval_deriv.eval(&tape_deriv, &[args.as_slice()]).unwrap();
        for (v, (a, b)) in args.iter().zip(out.iter().zip(out_deriv)) {
            let a = a.dx;
            let err = a - b;
            let err_frac = err / a.abs().max(b.abs());
            assert!(
                a == *b
                    || err < 1e-6
                    || err_frac < 1e-6
                    || (a.is_nan() && b.is_nan())
                    || v.is_nan(),
                "mismatch in '{}' at {v}: {a} != {b} ({err})",
                C::NAME,
            );
        }
    }

    pub fn test_binary<C: CanonicalBinaryOp>() {
        let args = test_args();

        let mut ctx = Context::new();
        let va = Var::new();
        let vb = Var::new();
        let a = ctx.var(va);
        let b = ctx.var(vb);

        let mut eval = VmFunction::new_grad_slice_eval();
        let mut eval_deriv = VmFunction::new_float_slice_eval();

        let node = C::build(&mut ctx, a, b);
        let shape = VmFunction::new(&ctx, node).unwrap();
        let tape = shape.grad_slice_tape(Default::default());

        let node_a_deriv = ctx.deriv(node, va).unwrap();
        let shape_a_deriv = VmFunction::new(&ctx, node_a_deriv).unwrap();
        let tape_a_deriv = shape_a_deriv.float_slice_tape(Default::default());

        let node_b_deriv = ctx.deriv(node, vb).unwrap();
        let shape_b_deriv = VmFunction::new(&ctx, node_b_deriv).unwrap();
        let tape_b_deriv = shape_b_deriv.float_slice_tape(Default::default());

        for rot in 0..args.len() {
            let mut rgsa = args.clone();
            rgsa.rotate_left(rot);

            let args_g = args
                .iter()
                .map(|v| Grad::new(*v, 1.0, 0.0, 0.0))
                .collect::<Vec<_>>();
            let rgsa_g = rgsa
                .iter()
                .map(|v| Grad::new(*v, 0.0, 1.0, 0.0))
                .collect::<Vec<_>>();

            let ia = shape.vars().get(&va).unwrap();
            let ib = shape.vars().get(&vb).unwrap();
            let mut vs = [[].as_slice(), [].as_slice()];
            vs[ia] = args_g.as_slice();
            vs[ib] = rgsa_g.as_slice();
            let out = eval.eval(&tape, &vs).unwrap();

            // Check symbolic differentiation results
            let mut vs = [args.as_slice(), args.as_slice()];
            if let Some(ia) = shape_a_deriv.vars().get(&va) {
                vs[ia] = args.as_slice();
            }
            if let Some(ib) = shape_a_deriv.vars().get(&vb) {
                vs[ib] = rgsa.as_slice();
            }
            let out_a_deriv =
                eval_deriv.eval(&tape_a_deriv, &vs).unwrap().to_vec();

            let mut vs = [args.as_slice(), args.as_slice()];
            if let Some(ia) = shape_b_deriv.vars().get(&va) {
                vs[ia] = args.as_slice();
            }
            if let Some(ib) = shape_b_deriv.vars().get(&vb) {
                vs[ib] = rgsa.as_slice();
            }
            let out_b_deriv = eval_deriv.eval(&tape_b_deriv, &vs).unwrap();

            for i in 0..out.len() {
                let v = out[i];
                let da = out_a_deriv[i];

                let a = args[i];
                let b = rgsa[i];

                let err = v.dx - da;
                let err_frac = err / da.abs().max(v.dx.abs());
                assert!(
                    v.dx == da
                        || err < 1e-6
                        || err_frac < 1e-6
                        || (v.dx.is_nan() && da.is_nan())
                        || v.v.is_nan(),
                    "mismatch in 'd {}(a, b) / da' at ({a}, {b}): \
                     {} != {da} ({err})",
                    C::NAME,
                    v.dx
                );

                let db = out_b_deriv[i];
                let err = v.dy - db;
                let err_frac = err / db.abs().max(v.dy.abs());
                assert!(
                    v.dy == db
                        || err < 1e-6
                        || err_frac < 1e-6
                        || (v.dy.is_nan() && db.is_nan())
                        || v.v.is_nan(),
                    "mismatch in 'd {}(a, b) / db' at ({a}, {b}): \
                     {} != {db} ({err})",
                    C::NAME,
                    v.dx
                );
            }
        }
    }
}

crate::all_unary_tests!(TestSymbolicDerivs);
crate::all_binary_tests!(TestSymbolicDerivs);
