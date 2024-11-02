//! Solver for systems of equations expressed as sets of [Function] objects
use crate::{
    eval::{BulkEvaluator, Function, Tape, TracingEvaluator},
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

/// Workspace for solvers
struct Solver<'a, F: Function> {
    /// Input parameters
    vars: &'a HashMap<Var, Parameter>,

    /// Tapes for bulk gradient evaluation of each constraint
    grad_tapes: Vec<<F::GradSliceEval as BulkEvaluator>::Tape>,

    /// Tapes for single-point evaluation of each constraint
    point_tapes: Vec<<F::PointEval as TracingEvaluator>::Tape>,

    /// Bulk gradient evaluator, for use in computing the Jacobian
    grad_eval: F::GradSliceEval,

    /// Single-point evaluator, for use in checking our current error
    point_eval: F::PointEval,

    /// Input data for use when calling the gradient bulk evaluator
    input_grad: Vec<Vec<Grad>>,

    /// Input data for use when calling the single-point evaluator
    input_point: Vec<f32>,

    /// Map from (free) variables to the index of their gradient
    ///
    /// We evaluate 3x gradients per sample, so for `grad_index = gi`, the
    /// relevant derivative will be `out[gi / 3].d(gi % 3)`
    grad_index: HashMap<Var, usize>,
}

impl<'a, F: Function> Solver<'a, F> {
    fn new(
        eqs: &'a [F],
        vars: &'a HashMap<Var, Parameter>,
    ) -> Result<Self, Error> {
        // Build our per-constraint
        let grad_tapes = eqs
            .iter()
            .map(|f| f.grad_slice_tape(Default::default()))
            .collect::<Vec<_>>();
        let point_tapes = eqs
            .iter()
            .map(|f| f.point_tape(Default::default()))
            .collect::<Vec<_>>();

        // Build a map from *free* variable to index of its gradient, since
        // we'll be using tightly-packed Vec everywhere here
        //
        // (We ignore the gradient of fixed variables)
        let grad_index: HashMap<Var, usize> = vars
            .iter()
            .filter(|(_v, p)| matches!(p, Parameter::Free(..)))
            .enumerate()
            .map(|(i, (v, _p))| (*v, i))
            .collect();

        // Build a scratch array with rows for each variable, and enough columns
        // to simultaneously compute all of the gradients that we need
        let input_grad =
            vec![
                vec![Grad::from(0f32); grad_index.len().div_ceil(3)];
                vars.len()
            ];
        let input_point = vec![0f32; vars.len()];

        Ok(Self {
            vars,
            grad_tapes,
            point_tapes,
            grad_eval: Default::default(),
            point_eval: Default::default(),
            grad_index,

            input_grad,
            input_point,
        })
    }

    /// Computes the Jacobian into `cur`
    ///
    /// # Panics
    /// If `jacobian` or `result` are an invalid size
    fn get_jacobian(
        &mut self,
        cur: &[f32],
        jacobian: &mut nalgebra::DMatrix<f32>,
        result: &mut nalgebra::DVector<f32>,
    ) -> Result<(), Error> {
        for (ti, tape) in self.grad_tapes.iter().enumerate() {
            // Update the values in the gradient evaluation array
            for (v, p) in self.vars {
                let Some(i) = tape.vars().get(v) else {
                    continue;
                };
                let Some(slice) = self.input_grad.get_mut(i) else {
                    return Err(Error::BadVarIndex(i, self.input_grad.len()));
                };
                match p {
                    Parameter::Free(..) => {
                        let gi = self.grad_index[v];
                        for (j, v) in slice.iter_mut().enumerate() {
                            *v = Grad::new(
                                cur[gi],
                                if j * 3 == gi { 1.0 } else { 0.0 },
                                if j * 3 + 1 == gi { 1.0 } else { 0.0 },
                                if j * 3 + 2 == gi { 1.0 } else { 0.0 },
                            );
                        }
                    }
                    Parameter::Fixed(f) => {
                        slice.fill(Grad::new(*f, 0.0, 0.0, 0.0));
                    }
                };
            }
            // Do the actual gradient evaluation
            let out = self.grad_eval.eval(tape, &self.input_grad)?;

            // Populate this row of the Jacobian
            for gi in 0..self.grad_index.len() {
                *jacobian.get_mut((ti, gi)).unwrap() = out[0][gi / 3].d(gi % 3);
            }
            result[ti] = out[0][0].v;
        }
        Ok(())
    }

    fn get_err(&mut self, cur: &[f32], delta: &[f32]) -> Result<f32, Error> {
        let mut err = 0f32;
        for tape in self.point_tapes.iter() {
            // Update the free values in the gradient evaluation array
            //
            // (we preloaded unit gradients and fixed values in the appropriate
            // locations, which don't change from evaluation to evaluation)
            for (v, p) in self.vars {
                let Some(i) = tape.vars().get(v) else {
                    continue;
                };
                let Some(f) = self.input_point.get_mut(i) else {
                    return Err(Error::BadVarIndex(i, self.input_point.len()));
                };
                match p {
                    Parameter::Free(..) => {
                        let gi = self.grad_index[v];
                        *f = cur[gi] - delta[gi];
                    }
                    Parameter::Fixed(p) => {
                        *f = *p;
                    }
                };
            }
            // Do the actual gradient evaluation
            let (out, _t) = self.point_eval.eval(tape, &self.input_point)?;
            err += out[0].powi(2); // TODO: consolidate into a single tape
        }
        Ok(err)
    }
}

/// Least-squares minimization on a set of functions
///
/// Returns a map from free variable to its final value
///
/// Minimization is accomplished using a relatively basic implementation of
/// the [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).
///
/// ## References
/// - [The Levenberg-Marquardt Algorithm (Ranganathan 2004)](http://ananth.in/docs/lmtut.pdf)
/// - [Basics on Continuous Optimization § Levenberg-Marquardt](https://www.brnt.eu/phd/node10.html#SECTION00622700000000000000)
/// - [Improvements to the Levenberg-Marquardt algorithm for nonlinear
///    least-squares minimization (Transtrum 2012)](https://arxiv.org/pdf/1201.5885)
pub fn solve<F: Function>(
    eqs: &[F],
    vars: &HashMap<Var, Parameter>,
) -> Result<HashMap<Var, f32>, Error> {
    let tapes = eqs
        .iter()
        .map(|f| f.grad_slice_tape(Default::default()))
        .collect::<Vec<_>>();

    // Current values for free variables
    let mut cur = HashMap::new();
    for (v, p) in vars {
        if let Parameter::Free(f) = *p {
            cur.insert(*v, f);
        }
    }

    let mut solver = Solver::new(eqs, vars)?;

    // Build an array of current values for each free variable
    let mut cur = vec![0f32; solver.grad_index.len()];
    for (v, i) in &solver.grad_index {
        let Parameter::Free(f) = vars[v] else {
            unreachable!();
        };
        cur[*i] = f;
    }

    // Working arrays for the current Jacobian and result
    let mut jacobian = nalgebra::DMatrix::repeat(tapes.len(), cur.len(), 0f32);
    let mut result = nalgebra::DVector::repeat(tapes.len(), 0f32);

    let mut damping = 1.0;
    let mut prev_err = f32::INFINITY;
    let mut err_buf = [0f32; 4];
    for i in 0.. {
        solver.get_jacobian(&cur, &mut jacobian, &mut result)?;

        // Early exit if we're done
        if result.iter().all(|v| *v == 0.0) {
            break;
        }

        let jt = jacobian.transpose();
        let jt_j = &jt * &jacobian;

        let jt_r = jt * &result;

        // TODO: be optimistic and evaluate the full gradient on the first
        // attempt, since it should usually succeed?
        let (err, step) = loop {
            let adjusted = &jt_j
                + damping * nalgebra::DMatrix::from_diagonal(&jt_j.diagonal());

            let delta = adjusted
                .svd(true, true)
                .solve(&jt_r, f32::EPSILON)
                .map_err(Error::SingularMatrix)?;

            let err = solver.get_err(&cur, delta.as_slice())?;
            if err > prev_err {
                // Keep going in this inner loop, taking smaller steps
                damping *= 1.5;
            } else {
                // We found a good step size, so reduce damping
                damping /= 3.0;
                break (err, delta);
            }
        };

        // Update our current position, checking whether it actually changed
        // (i.e. whether our steps are below the floating-point epsilon)
        //
        // TODO: improve exit criteria?
        let mut changed = false;
        for gi in 0..solver.grad_index.len() {
            let prev = cur[gi];
            cur[gi] -= step[gi];
            changed |= prev != cur[gi];
        }
        err_buf[i % err_buf.len()] = err;
        if !changed
            || err == 0.0
            || damping == 0.0
            || err_buf.iter().all(|e| *e == err_buf[0])
        {
            break;
        }
        prev_err = err;
    }

    // Return the new "current" values, which are our optimized position
    let out = solver
        .grad_index
        .into_iter()
        .map(|(v, i)| (v, cur[i]))
        .collect();
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
    use approx::{assert_relative_eq, relative_eq};

    #[test]
    fn basic_solver() {
        let eqn = Tree::x() + Tree::y();
        let mut ctx = Context::new();
        let root = ctx.import(&eqn);

        let f = VmFunction::new(&ctx, &[root]).unwrap();
        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));
        values.insert(Var::Y, Parameter::Fixed(-1.0));
        let sol = solve(&[f], &values).unwrap();
        assert_eq!(sol.len(), 1);
        assert_relative_eq!(sol[&Var::X], 1.0);
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

        let f = VmFunction::new(&ctx, &[root]).unwrap();
        let mut values = HashMap::new();
        for (i, &v) in vs.iter().enumerate() {
            values.insert(v, Parameter::Free(i as f32));
        }
        let sol = solve(&[f], &values).unwrap();
        assert_eq!(sol.len(), 4);
        let mut out = 0.0;
        for v in &vs {
            out += sol[v];
        }
        assert_relative_eq!(out, 0.0);
    }

    #[test]
    fn four_vars_independent() {
        let vs = (0..4).map(|_| Var::new()).collect::<Vec<Var>>();
        let mut eqns = vec![];
        let mut ctx = Context::new();
        for (i, &v) in vs.iter().enumerate() {
            let eqn = Tree::from(v) - Tree::from(i as f32);
            let root = ctx.import(&eqn);
            let f = VmFunction::new(&ctx, &[root]).unwrap();
            eqns.push(f);
        }

        let mut values = HashMap::new();
        for (i, &v) in vs.iter().enumerate() {
            values.insert(v, Parameter::Free(i as f32 * 2.0));
        }
        let sol = solve(&eqns, &values).unwrap();
        assert_eq!(sol.len(), 4);
        for (i, v) in vs.iter().enumerate() {
            assert_relative_eq!(i as f32, sol[v]);
        }
    }

    #[test]
    fn xy_nonlinear() {
        let constraints = vec![
            (Tree::x() * 2 + Tree::y() * 3) * (Tree::x() - Tree::y()) - 2,
            Tree::x() * 3 + Tree::y() - 5,
        ];
        let mut ctx = Context::new();
        let eqns = constraints
            .into_iter()
            .map(|c| {
                let root = ctx.import(&c);
                VmFunction::new(&ctx, &[root]).unwrap()
            })
            .collect::<Vec<_>>();

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));
        values.insert(Var::Y, Parameter::Free(0.0));
        let sol = solve(&eqns, &values).unwrap();

        let x = sol[&Var::X];
        let y = sol[&Var::Y];

        assert_relative_eq!((x * 2.0 + y * 3.0) * (x - y), 2.0);
        assert_relative_eq!(x * 3.0 + y, 5.0);
    }

    #[test]
    fn one_var_no_solution() {
        // Solve for X == 1 and X == 2 simultaneously
        let constraints = vec![Tree::x() - 1.0, Tree::x() - 2.0];

        let mut ctx = Context::new();
        let eqns = constraints
            .into_iter()
            .map(|c| {
                let root = ctx.import(&c);
                VmFunction::new(&ctx, &[root]).unwrap()
            })
            .collect::<Vec<_>>();

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));

        let sol = solve(&eqns, &values).unwrap();

        let x = sol[&Var::X];
        assert_relative_eq!(x, 1.5);
    }

    #[test]
    fn solve_banana() {
        // See https://en.wikipedia.org/wiki/Rosenbrock_function
        let a = 1f32;
        let b = 100f32;
        let constraints = [a - Tree::x(), b * (Tree::y() - Tree::x().square())];

        let mut ctx = Context::new();
        let eqns = constraints
            .into_iter()
            .map(|c| {
                let root = ctx.import(&c);
                VmFunction::new(&ctx, &[root]).unwrap()
            })
            .collect::<Vec<_>>();

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));
        values.insert(Var::Y, Parameter::Free(0.0));
        let sol = solve(&eqns, &values).unwrap();
        assert_relative_eq!(sol[&Var::X], 1.0);
        assert_relative_eq!(sol[&Var::Y], 1.0);

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(1.0));
        values.insert(Var::Y, Parameter::Free(1.0));
        let sol = solve(&eqns, &values).unwrap();
        assert_relative_eq!(sol[&Var::X], 1.0);
        assert_relative_eq!(sol[&Var::Y], 1.0);
    }

    #[test]
    fn solve_circle() {
        let t = (Tree::x().square() + Tree::y().square()).sqrt();
        let mut ctx = Context::new();
        let root = ctx.import(&t);
        let eqn = VmFunction::new(&ctx, &[root]).unwrap();
        let eqns = [eqn];

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(0.0));
        values.insert(Var::Y, Parameter::Free(0.0));
        let sol = solve(&eqns, &values).unwrap();
        assert_relative_eq!(sol[&Var::X], 0.0);
        assert_relative_eq!(sol[&Var::Y], 0.0);

        let mut values = HashMap::new();
        values.insert(Var::X, Parameter::Free(1.0));
        values.insert(Var::Y, Parameter::Free(1.5));
        let sol = solve(&eqns, &values).unwrap();
        assert_relative_eq!(sol[&Var::X], 0.0);
        assert_relative_eq!(sol[&Var::Y], 0.0);
    }

    fn one_linear(n: usize) {
        // Build a random matrix of our solutions
        let mut values = nalgebra::DVector::<f32>::zeros(n);
        for v in values.iter_mut() {
            *v = rand::random();
        }

        let vars = (0..n).map(|_| Var::new()).collect::<Vec<_>>();
        let trees = vars.iter().map(|v| Tree::from(*v)).collect::<Vec<_>>();

        let mut mat = nalgebra::DMatrix::<f32>::zeros(n, n);
        for v in mat.iter_mut() {
            *v = rand::random();
        }

        let sol = &mat * &values;

        let mut ctx = Context::new();
        let mut eqns = vec![];
        for row in 0..n {
            let mut out = Tree::from(-sol[row]);
            for (col, t) in trees.iter().enumerate() {
                out += *mat.get((row, col)).unwrap() * t.clone();
            }
            let root = ctx.import(&out);
            let f = VmFunction::new(&ctx, &[root]).unwrap();
            eqns.push(f);
        }

        let params = vars.iter().map(|v| (*v, Parameter::Free(0.0))).collect();
        let out = solve(&eqns, &params).unwrap();

        // It's possible for there to be multiple solutions here, so we'll check
        // the actual equations.
        for i in 0..n {
            values[i] = out[&vars[i]];
        }
        let sol2 = &mat * &values;
        let err = (&sol - &sol2).norm_squared();
        assert!(err < 1e-3, "error {err} is too large");
        for (a, b) in sol.iter().zip(sol2.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-2);
        }
    }

    #[test]
    fn small_linear() {
        for _ in 0..1000 {
            one_linear(2);
        }
    }

    #[test]
    fn medium_linear() {
        for _ in 0..1000 {
            one_linear(10);
        }
    }

    #[test]
    fn big_linear() {
        for _ in 0..50 {
            one_linear(50);
        }
    }

    fn one_quadratic(n: usize) -> bool {
        let m: usize = n * n + n;

        // Build a random matrix of our solutions
        let mut values = nalgebra::DVector::<f32>::zeros(n);
        for v in values.iter_mut() {
            *v = rand::random();
        }

        // Build a column vector of [a b c ... aa ab ac ... ba bb ...]^T
        let mut col = nalgebra::DVector::<f32>::zeros(m);
        col.rows_range_mut(..n).copy_from(&values);
        for i in 0..n {
            for j in 0..n {
                let index = i * n + j + n;
                col[index] = values[i] * values[j];
            }
        }

        let vars = (0..n).map(|_| Var::new()).collect::<Vec<_>>();
        let trees = vars.iter().map(|v| Tree::from(*v)).collect::<Vec<_>>();

        let mut mat = nalgebra::DMatrix::<f32>::zeros(n, m);
        for v in mat.iter_mut() {
            *v = rand::random();
        }

        let sol = &mat * &col;

        let mut ctx = Context::new();
        let mut eqns = vec![];
        for row in 0..n {
            let mut out = Tree::from(-sol[row]);
            for (col, t) in trees.iter().enumerate() {
                out += *mat.get((row, col)).unwrap() * t.clone();
            }
            for i in 0..n {
                for j in 0..n {
                    let index = i * n + j + n;
                    out += *mat.get((row, index)).unwrap()
                        * trees[i].clone()
                        * trees[j].clone();
                }
            }
            let root = ctx.import(&out);
            let f = VmFunction::new(&ctx, &[root]).unwrap();
            eqns.push(f);
        }

        let params = vars.iter().map(|v| (*v, Parameter::Free(0.5))).collect();
        let out = solve(&eqns, &params).unwrap();

        // It's possible for there to be multiple solutions here, so we'll check
        // the actual equations.
        for i in 0..n {
            col[i] = out[&vars[i]];
            for j in 0..n {
                let index = i * n + j + n;
                col[index] = out[&vars[i]] * out[&vars[j]];
            }
        }
        let sol2 = &mat * &col;
        let err = (&sol - &sol2).norm_squared();
        if err >= 1e-3 {
            return false;
        }
        for (a, b) in sol.iter().zip(sol2.iter()) {
            if !relative_eq!(a, b, epsilon = 1e-2) {
                return false;
            }
        }
        true
    }

    // Quadratic functions can get trapped in local minima, so we only require a
    // certain percent to succeed (hard-coded to 90% right now)
    fn many_quadratic(size: usize, count: usize) {
        let mut okay = 0;
        for _ in 0..count {
            if one_quadratic(size) {
                okay += 1;
            }
        }
        assert!(
            okay >= count * 9 / 10,
            "too many failures: {okay} / {count}"
        );
    }

    #[test]
    fn small_quadratic() {
        many_quadratic(2, 1000);
    }

    #[test]
    fn medium_quadratic() {
        many_quadratic(5, 100);
    }

    #[test]
    fn large_quadratic() {
        many_quadratic(10, 50);
    }
}
