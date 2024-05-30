//! Solver for systems of equations expressed as sets of [`Function`]s
use crate::{
    eval::{BulkEvaluator, Function, Tape},
    types::Grad,
    var::Var,
    Error,
};
use std::collections::HashMap;

/// Input parameter to the solver
#[derive(Copy, Clone, Debug)]
pub enum Parameter {
    /// Free variable with the given starting position
    Free(f32),
    /// Fixed variable at the given value
    Fixed(f32),
}

/// Least-squares minimization on a set of functions
///
/// Returns a map from free variable to their final value
pub fn solve<F: Function>(
    eqs: &[F],
    vars: &HashMap<Var, Parameter>,
) -> Result<HashMap<Var, f32>, Error> {
    // Levenberg–Marquardt algorithm

    let tapes = eqs
        .iter()
        .map(|f| f.grad_slice_tape(Default::default()))
        .collect::<Vec<_>>();
    let mut eval = F::new_grad_slice_eval();

    // Current values for free variables
    let mut cur = HashMap::new();
    for (v, p) in vars {
        if let Parameter::Free(f) = *p {
            cur.insert(*v, f);
        }
    }

    // Build a map from free variable to index of its gradient, since we'll be
    // using tightly-packed Vec everywhere here
    let grad_index: HashMap<Var, usize> = vars
        .iter()
        .filter(|(_v, p)| matches!(p, Parameter::Free(..)))
        .enumerate()
        .map(|(i, (v, _p))| (*v, i))
        .collect();

    // Build an array of current values for each free variable
    let mut cur = vec![0f32; grad_index.len()];
    for (v, i) in &grad_index {
        let Parameter::Free(f) = vars[v] else {
            unreachable!();
        };
        cur[*i] = f;
    }

    // Build a scratch array with rows for each variable, and enough columns to
    // simultaneously compute all of the gradients that we need
    let mut scratch =
        vec![vec![Grad::from(0f32); cur.len().div_ceil(3)]; vars.len()];

    let mut jacobian = nalgebra::DMatrix::repeat(tapes.len(), cur.len(), 0f32);
    let mut result = nalgebra::DVector::repeat(tapes.len(), 0f32);

    for _ in 0..100 {
        for (ti, tape) in tapes.iter().enumerate() {
            // Build the evaluation data array
            // TODO: set up gradients once then only change values?
            for (v, p) in vars {
                let (gi, f) = match p {
                    Parameter::Free(..) => {
                        let gi = grad_index[v];
                        (Some(gi), cur[gi])
                    }
                    Parameter::Fixed(f) => (None, *f),
                };
                let var_map = tape.vars();
                let Some(i) = var_map.get(v) else {
                    // TODO split into independent subproblems?
                    continue;
                };
                if i >= scratch.len() {
                    return Err(Error::BadVarIndex(i, scratch.len()));
                }
                let slice = &mut scratch[var_map[v]];
                if let Some(gi) = gi {
                    for (j, v) in slice.iter_mut().enumerate() {
                        *v = Grad::new(
                            f,
                            if j * 3 == gi { 1.0 } else { 0.0 },
                            if j * 3 + 1 == gi { 1.0 } else { 0.0 },
                            if j * 3 + 2 == gi { 1.0 } else { 0.0 },
                        );
                    }
                } else {
                    slice.fill(Grad::new(f, 0.0, 0.0, 0.0));
                }
            }
            let out = eval.eval(tape, &scratch)?;

            // Populate this row of the Jacobian
            for gi in 0..grad_index.len() {
                *jacobian.get_mut((ti, gi)).unwrap() = out[gi / 3].d(gi % 3);
            }
            result[ti] = out[0].v;
        }
        // TODO: calculate the next step and update `cur`
        // TODO: determine exit critera for breaking out of the loop
        //
        let jt = jacobian.transpose();
        let jt_j = &jt * &jacobian;

        let err = jt * &result;

        let jt_j_i = (&jt_j
            + nalgebra::DMatrix::from_diagonal(&jt_j.diagonal()))
        .try_inverse()
        .unwrap();

        let delta = jt_j_i * err;

        for gi in 0..grad_index.len() {
            cur[gi] -= delta[gi];
        }
    }
    println!("got result \n{result}");

    // Return the new "current" values, which are our optimized position
    let out = grad_index.into_iter().map(|(v, i)| (v, cur[i])).collect();
    println!("solved to {out:?}");
    Ok(out)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        context::{Context, Tree},
        eval::MathFunction,
        vm::VmFunction,
    };

    #[test]
    fn basic_solver() {
        let eqn = Tree::x() + Tree::y();
        let mut ctx = Context::new();
        let root = ctx.import(&eqn);

        let f = VmFunction::new(&ctx, root).unwrap();
        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));
        values.insert(Var::Y, Parameter::Fixed(-1.0));
        solve(&[f], &values).unwrap();
    }

    #[test]
    fn four_vars_at_once() {
        let vs = (0..4).map(|_| Var::new()).collect::<Vec<Var>>();
        let mut root = Tree::from(vs[0]);
        for v in &vs[1..] {
            root += Tree::from(*v);
        }
        let mut ctx = Context::new();
        let root = ctx.import(&root);

        let f = VmFunction::new(&ctx, root).unwrap();
        let mut values = HashMap::new();
        for (i, &v) in vs.iter().enumerate() {
            values.insert(v, Parameter::Free(i as f32));
        }
        solve(&[f], &values).unwrap();
    }

    #[test]
    fn four_vars_independent() {
        let vs = (0..4).map(|_| Var::new()).collect::<Vec<Var>>();
        let mut eqns = vec![];
        let mut ctx = Context::new();
        for (i, &v) in vs.iter().enumerate() {
            let eqn = Tree::from(v) - Tree::from(i as f32);
            let root = ctx.import(&eqn);
            let f = VmFunction::new(&ctx, root).unwrap();
            eqns.push(f);
        }

        let mut values = HashMap::new();
        for (i, &v) in vs.iter().enumerate() {
            values.insert(v, Parameter::Free(i as f32 * 2.0));
        }
        solve(&eqns, &values).unwrap();
    }
}
