//! Interval evaluation tests
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.

use super::{
    bind_xy, bind_xyz, build_stress_fn, test_args, test_args_n,
    CanonicalBinaryOp, CanonicalUnaryOp,
};
use crate::{
    context::Context,
    eval::{Function, MathFunction, Tape, TracingEvaluator},
    shape::EzShape,
    types::Interval,
    var::Var,
    vm::{Choice, VmShape},
};

/// Helper struct to put constrains on our `Shape` object
pub struct TestInterval<F>(std::marker::PhantomData<*const F>);

impl<F> TestInterval<F>
where
    for<'a> F: Function + MathFunction,
    <F as Function>::Trace: AsRef<[Choice]>,
{
    pub fn test_interval() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape = F::new(&ctx, &[x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 5.0].into()]).unwrap().0[0],
            [1.0, 5.0].into()
        );

        let shape = F::new(&ctx, &[y]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[2.0, 3.0].into()]).unwrap().0[0],
            [2.0, 3.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[4.0, 5.0].into()]).unwrap().0[0],
            [4.0, 5.0].into()
        );
    }

    pub fn test_i_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let shape = F::new(&ctx, &[abs_x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 5.0].into()]).unwrap().0[0],
            [1.0, 5.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-2.0, 5.0].into()]).unwrap().0[0],
            [0.0, 5.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, 5.0].into()]).unwrap().0[0],
            [0.0, 6.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, -1.0].into()]).unwrap().0[0],
            [1.0, 6.0].into()
        );

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let shape = F::new(&ctx, &[sum]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &vs([0.0, 1.0], [0.0, 1.0])).unwrap().0[0],
            [0.0, 2.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([1.0, 5.0], [-2.0, 3.0])).unwrap().0[0],
            [1.0, 8.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([1.0, 5.0], [-4.0, 3.0])).unwrap().0[0],
            [1.0, 9.0].into()
        );
    }

    pub fn test_i_add_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let v = ctx.add(x, 0.5).unwrap();
        let out = ctx.abs(v).unwrap();

        let shape = F::new(&ctx, &[out]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();

        assert_eq!(
            eval.eval(&tape, &[[-1.0, 1.0].into()]).unwrap().0[0],
            [0.0, 1.5].into()
        );
    }

    pub fn test_i_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let shape = F::new(&ctx, &[sqrt_x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[0.0, 4.0].into()]).unwrap().0[0],
            [0.0, 2.0].into()
        );

        // Even a partial negative returns a NAN interval
        let nanan = eval.eval(&tape, &[[-2.0, 4.0].into()]).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        // Full negatives are right out
        let nanan = eval.eval(&tape, &[[-2.0, -1.0].into()]).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let v = eval.eval(&tape, &[[f32::NAN; 2].into()]).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let shape = F::new(&ctx, &[sqrt_x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[0.0, 4.0].into()]).unwrap().0[0],
            [0.0, 16.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[2.0, 4.0].into()]).unwrap().0[0],
            [4.0, 16.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-2.0, 4.0].into()]).unwrap().0[0],
            [0.0, 16.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, -2.0].into()]).unwrap().0[0],
            [4.0, 36.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, 1.0].into()]).unwrap().0[0],
            [0.0, 36.0].into()
        );

        let v = eval.eval(&tape, &[[f32::NAN; 2].into()]).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_sin() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();
        let shape = F::new(&ctx, &[s]).unwrap();
        let tape = shape.interval_tape(Default::default());

        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [-1.0, 1.0].into()
        );

        let y = ctx.y();
        let y = ctx.mul(y, 2.0).unwrap();
        let s = ctx.sin(y).unwrap();
        let s = ctx.add(x, s).unwrap();
        let shape = F::new(&ctx, &[s]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);

        assert_eq!(
            eval.eval(&tape, &vs([0.0, 3.0], [0.0, 0.0])).unwrap().0[0],
            [-1.0, 4.0].into()
        );
    }

    pub fn test_i_neg() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let neg_x = ctx.neg(x).unwrap();

        let shape = F::new(&ctx, &[neg_x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [-1.0, 0.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[0.0, 4.0].into()]).unwrap().0[0],
            [-4.0, 0.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[2.0, 4.0].into()]).unwrap().0[0],
            [-4.0, -2.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-2.0, 4.0].into()]).unwrap().0[0],
            [-4.0, 2.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, -2.0].into()]).unwrap().0[0],
            [2.0, 6.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[-6.0, 1.0].into()]).unwrap().0[0],
            [-1.0, 6.0].into()
        );

        let v = eval.eval(&tape, &[[f32::NAN; 2].into()]).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_not() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let not_x = ctx.not(x).unwrap();

        let shape = F::new(&ctx, &[not_x]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[-5.0, 0.0].into()]).unwrap().0[0],
            [0.0, 1.0].into()
        );
    }

    pub fn test_i_mul() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let shape = F::new(&ctx, &[mul]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &vs([0.0, 1.0], [0.0, 1.0])).unwrap().0[0],
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([0.0, 1.0], [0.0, 2.0])).unwrap().0[0],
            [0.0, 2.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-2.0, 1.0], [0.0, 1.0])).unwrap().0[0],
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-2.0, -1.0], [-5.0, -4.0])).unwrap().0[0],
            [4.0, 10.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-3.0, -1.0], [-2.0, 6.0])).unwrap().0[0],
            [-18.0, 6.0].into()
        );

        let v = eval.eval(&tape, &vs([f32::NAN; 2], [0.0, 1.0])).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let v = eval.eval(&tape, &vs([0.0, 1.0], [f32::NAN; 2])).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let mul = ctx.mul(x, 2.0).unwrap();
        let shape = F::new(&ctx, &[mul]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [0.0, 2.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 2.0].into()]).unwrap().0[0],
            [2.0, 4.0].into()
        );

        let mul = ctx.mul(x, -3.0).unwrap();
        let shape = F::new(&ctx, &[mul]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [-3.0, 0.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 2.0].into()]).unwrap().0[0],
            [-6.0, -3.0].into()
        );
    }

    pub fn test_i_sub() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let shape = F::new(&ctx, &[sub]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &vs([0.0, 1.0], [0.0, 1.0])).unwrap().0[0],
            [-1.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([0.0, 1.0], [0.0, 2.0])).unwrap().0[0],
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-2.0, 1.0], [0.0, 1.0])).unwrap().0[0],
            [-3.0, 1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-2.0, -1.0], [-5.0, -4.0])).unwrap().0[0],
            [2.0, 4.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &vs([-3.0, -1.0], [-2.0, 6.0])).unwrap().0[0],
            [-9.0, 1.0].into()
        );
    }

    pub fn test_i_sub_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sub = ctx.sub(x, 2.0).unwrap();
        let shape = F::new(&ctx, &[sub]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [-2.0, -1.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 2.0].into()]).unwrap().0[0],
            [-1.0, 0.0].into()
        );

        let sub = ctx.sub(-3.0, x).unwrap();
        let shape = F::new(&ctx, &[sub]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        assert_eq!(
            eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0],
            [-4.0, -3.0].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 2.0].into()]).unwrap().0[0],
            [-5.0, -4.0].into()
        );
    }

    pub fn test_i_recip() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let shape = F::new(&ctx, &[recip]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();

        let nanan = eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval(&tape, &[[-1.0, 0.0].into()]).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval(&tape, &[[-2.0, 3.0].into()]).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        assert_eq!(
            eval.eval(&tape, &[[-2.0, -1.0].into()]).unwrap().0[0],
            [-1.0, -0.5].into()
        );
        assert_eq!(
            eval.eval(&tape, &[[1.0, 2.0].into()]).unwrap().0[0],
            [0.5, 1.0].into()
        );
    }

    pub fn test_i_div() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let div = ctx.div(x, y).unwrap();
        let shape = F::new(&ctx, &[div]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();

        let nanan =
            eval.eval(&tape, &vs([0.0, 1.0], [-1.0, 1.0])).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan =
            eval.eval(&tape, &vs([0.0, 1.0], [-2.0, 0.0])).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval(&tape, &vs([0.0, 1.0], [0.0, 4.0])).unwrap().0[0];
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let out = eval.eval(&tape, &vs([-1.0, 0.0], [1.0, 2.0])).unwrap().0[0];
        assert_eq!(out, [-1.0, 0.0].into());

        let out =
            eval.eval(&tape, &vs([-1.0, 4.0], [-1.0, -0.5])).unwrap().0[0];
        assert_eq!(out, [-8.0, 2.0].into());

        let out = eval.eval(&tape, &vs([1.0, 4.0], [-1.0, -0.5])).unwrap().0[0];
        assert_eq!(out, [-8.0, -1.0].into());

        let out = eval.eval(&tape, &vs([-1.0, 4.0], [0.5, 1.0])).unwrap().0[0];
        assert_eq!(out, [-2.0, 8.0].into());

        let v = eval.eval(&tape, &vs([f32::NAN; 2], [0.0, 1.0])).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let v = eval.eval(&tape, &vs([0.0, 1.0], [f32::NAN; 2])).unwrap().0[0];
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = F::new(&ctx, &[min]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        let (r, data) = eval.eval(&tape, &vs([0.0, 1.0], [0.5, 1.5])).unwrap();
        assert_eq!(r[0], [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) = eval.eval(&tape, &vs([0.0, 1.0], [2.0, 3.0])).unwrap();
        assert_eq!(r[0], [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval.eval(&tape, &vs([2.0, 3.0], [0.0, 1.0])).unwrap();
        assert_eq!(r[0], [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (v, data) =
            eval.eval(&tape, &vs([f32::NAN; 2], [0.0, 1.0])).unwrap();
        assert!(v[0].lower().is_nan());
        assert!(v[0].upper().is_nan());
        assert!(data.is_none());

        let (v, data) =
            eval.eval(&tape, &vs([0.0, 1.0], [f32::NAN; 2])).unwrap();
        assert!(v[0].lower().is_nan());
        assert!(v[0].upper().is_nan());
        assert!(data.is_none());
    }

    pub fn test_i_min_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let shape = F::new(&ctx, &[min]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        let (r, data) = eval.eval(&tape, &[[0.0, 1.0].into()]).unwrap();
        assert_eq!(r[0], [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) = eval.eval(&tape, &[[-1.0, 0.0].into()]).unwrap();
        assert_eq!(r[0], [-1.0, 0.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval.eval(&tape, &[[2.0, 3.0].into()]).unwrap();
        assert_eq!(r[0], [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);
    }

    pub fn test_i_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let shape = F::new(&ctx, &[max]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        let (r, data) = eval.eval(&tape, &vs([0.0, 1.0], [0.5, 1.5])).unwrap();
        assert_eq!(r[0], [0.5, 1.5].into());
        assert!(data.is_none());

        let (r, data) = eval.eval(&tape, &vs([0.0, 1.0], [2.0, 3.0])).unwrap();
        assert_eq!(r[0], [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval.eval(&tape, &vs([2.0, 3.0], [0.0, 1.0])).unwrap();
        assert_eq!(r[0], [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (v, data) =
            eval.eval(&tape, &vs([f32::NAN; 2], [0.0, 1.0])).unwrap();
        assert!(v[0].lower().is_nan());
        assert!(v[0].upper().is_nan());
        assert!(data.is_none());

        let (v, data) =
            eval.eval(&tape, &vs([0.0, 1.0], [f32::NAN; 2])).unwrap();
        assert!(v[0].lower().is_nan());
        assert!(v[0].upper().is_nan());
        assert!(data.is_none());

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let shape = F::new(&ctx, &[max_xy_z]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xyz(&tape);
        let mut eval = F::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, &vs([2.0, 3.0], [0.0, 1.0], [4.0, 5.0]))
            .unwrap();
        assert_eq!(r[0], [4.0, 5.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Right]);

        let (r, data) = eval
            .eval(&tape, &vs([2.0, 3.0], [0.0, 1.0], [1.0, 4.0]))
            .unwrap();
        assert_eq!(r[0], [2.0, 4.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Both]);

        let (r, data) = eval
            .eval(&tape, &vs([2.0, 3.0], [0.0, 1.0], [1.0, 1.5]))
            .unwrap();
        assert_eq!(r[0], [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Left]);
    }

    pub fn test_i_and() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let v = ctx.and(x, y).unwrap();

        let shape = F::new(&ctx, &[v]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        let (r, trace) =
            eval.eval(&tape, &vs([0.0, 0.0], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [0.0, 0.0].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) =
            eval.eval(&tape, &vs([-1.0, -0.2], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [-1.0, 3.0].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) =
            eval.eval(&tape, &vs([0.2, 1.3], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [-1.0, 3.0].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) =
            eval.eval(&tape, &vs([-0.2, 1.3], [1.0, 3.0])).unwrap();
        assert_eq!(r[0], [0.0, 3.0].into());
        assert!(trace.is_none()); // can't simplify
    }

    pub fn test_i_or() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let v = ctx.or(x, y).unwrap();

        let shape = F::new(&ctx, &[v]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        let (r, trace) =
            eval.eval(&tape, &vs([0.0, 0.0], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [-1.0, 3.0].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

        let (r, trace) =
            eval.eval(&tape, &vs([-1.0, -0.2], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [-1.0, -0.2].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) =
            eval.eval(&tape, &vs([0.2, 1.3], [-1.0, 3.0])).unwrap();
        assert_eq!(r[0], [0.2, 1.3].into());
        assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

        let (r, trace) =
            eval.eval(&tape, &vs([-0.2, 1.3], [1.0, 3.0])).unwrap();
        assert_eq!(r[0], [-0.2, 3.0].into());
        assert!(trace.is_none());
    }

    pub fn test_i_simplify() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let shape = F::new(&ctx, &[min]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        let (out, data) = eval.eval(&tape, &[[0.0, 2.0].into()]).unwrap();
        assert_eq!(out[0], [0.0, 1.0].into());
        assert!(data.is_none());

        let (out, data) = eval.eval(&tape, &[[0.0, 0.5].into()]).unwrap();
        assert_eq!(out[0], [0.0, 0.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (out, data) = eval.eval(&tape, &[[1.5, 2.5].into()]).unwrap();
        assert_eq!(out[0], [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let max = ctx.max(x, 1.0).unwrap();
        let shape = F::new(&ctx, &[max]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        let (out, data) = eval.eval(&tape, &[[0.0, 2.0].into()]).unwrap();
        assert_eq!(out[0], [1.0, 2.0].into());
        assert!(data.is_none());

        let (out, data) = eval.eval(&tape, &[[0.0, 0.5].into()]).unwrap();
        assert_eq!(out[0], [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (out, data) = eval.eval(&tape, &[[1.5, 2.5].into()]).unwrap();
        assert_eq!(out[0], [1.5, 2.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_simplify_conditional() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();

        let if_else = ctx.if_nonzero_else(x, y, z).unwrap();
        let shape = F::new(&ctx, &[if_else]).unwrap();

        let tape = shape.interval_tape(Default::default());
        let vs = bind_xyz(&tape);
        let mut eval = F::new_interval_eval();
        let (out, data) = eval
            .eval(&tape, &vs([-1.0, 2.0], [1.0, 2.0], [3.0, 4.0]))
            .unwrap();

        // Alas, we lose the information that the conditional is correlated, so
        // the interval naively must include 0
        assert_eq!(out[0], [0.0, 4.0].into());
        assert!(data.is_none());

        // Confirm that simplification of the right side works
        let (out, data) = eval
            .eval(&tape, &vs([0.0, 0.0], [1.0, 2.0], [3.0, 4.0]))
            .unwrap();
        assert_eq!(out[0], [3.0, 4.0].into());
        let s_z = shape
            .simplify(
                data.expect("must have trace"),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let t_z = s_z.interval_tape(Default::default());
        let (out, data) = eval
            .eval(&t_z, &vs([-1.0, 1.0], [1.0, 2.0], [5.0, 6.0]))
            .unwrap();
        assert!(s_z.size() < shape.size());
        assert_eq!(out[0], [5.0, 6.0].into());
        assert!(data.is_none());

        let (out, data) = eval
            .eval(&tape, &vs([1.0, 3.0], [1.0, 2.0], [3.0, 4.0]))
            .unwrap();
        assert_eq!(out[0], [1.0, 2.0].into());
        assert!(data.is_some());
        let s_y = shape
            .simplify(
                data.expect("must have trace"),
                Default::default(),
                &mut Default::default(),
            )
            .unwrap();
        let t_y = s_y.interval_tape(Default::default());
        let (out, data) = eval
            .eval(&t_y, &vs([-1.0, 1.0], [1.0, 4.0], [5.0, 6.0]))
            .unwrap();
        assert!(s_y.size() < shape.size());
        assert_eq!(out[0], [1.0, 4.0].into());
        assert!(data.is_none());

        assert_eq!(s_y.size(), s_z.size())
    }

    pub fn test_i_max_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let max = ctx.max(x, 1.0).unwrap();

        let shape = F::new(&ctx, &[max]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let mut eval = F::new_interval_eval();
        let (r, data) = eval.eval(&tape, &[[0.0, 2.0].into()]).unwrap();
        assert_eq!(r[0], [1.0, 2.0].into());
        assert!(data.is_none());

        let (r, data) = eval.eval(&tape, &[[-1.0, 0.0].into()]).unwrap();
        assert_eq!(r[0], [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval.eval(&tape, &[[2.0, 3.0].into()]).unwrap();
        assert_eq!(r[0], [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_compare() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let c = ctx.compare(x, y).unwrap();

        let shape = F::new(&ctx, &[c]).unwrap();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xy(&tape);
        let mut eval = F::new_interval_eval();
        let (out, _trace) = eval.eval(&tape, &vs(-5.0, -6.0)).unwrap();
        assert_eq!(out[0], Interval::from(1f32));
    }

    pub fn test_i_stress_n(depth: usize) {
        let (ctx, node) = build_stress_fn(depth);

        // Pick an input slice that's guaranteed to be > 1 SIMD register
        let args = (0..32).map(|i| i as f32 / 32f32).collect::<Vec<f32>>();
        let x: Vec<_> = args
            .iter()
            .zip(args.iter())
            .map(|(a, b)| Interval::new(*a, *a + *b))
            .collect();
        let y: Vec<_> = x[1..].iter().chain(&x[0..1]).cloned().collect();
        let z: Vec<_> = x[2..].iter().chain(&x[0..2]).cloned().collect();

        let shape = F::new(&ctx, &[node]).unwrap();
        let mut eval = F::new_interval_eval();
        let tape = shape.interval_tape(Default::default());
        let vs = bind_xyz(&tape);

        let mut out = vec![];
        for i in 0..args.len() {
            out.push(eval.eval(&tape, &vs(x[i], y[i], z[i])).unwrap().0[0]);
        }

        // Compare against the VmShape evaluator as a baseline.  It's possible
        // that S is also a VmShape, but this comparison isn't particularly
        // expensive, so we'll do it regardless.
        let shape = VmShape::new(&ctx, node).unwrap();
        let mut eval = VmShape::new_interval_eval();
        let tape = shape.ez_interval_tape();

        let mut cmp = vec![];
        for i in 0..args.len() {
            cmp.push(eval.eval(&tape, x[i], y[i], z[i]).unwrap().0);
        }

        for (a, b) in out.iter().zip(cmp.iter()) {
            a.compare_eq(*b)
        }
    }

    pub fn test_i_stress() {
        for n in [4, 8, 12, 16, 32] {
            Self::test_i_stress_n(n);
        }
    }

    pub fn interval_test_args() -> Vec<Interval> {
        let args = test_args_n(8);
        let mut out = vec![];
        for &lower in &args {
            for &size in &args {
                if size >= 0.0 {
                    out.push(Interval::new(lower, lower + size));
                }
            }
        }
        out.push(Interval::new(f32::NAN, f32::NAN));
        out
    }

    pub fn test_unary<C: CanonicalUnaryOp>() {
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let mut eval = F::new_interval_eval();

        let v = ctx.var(Var::new());
        let node = C::build(&mut ctx, v);

        let shape = F::new(&ctx, &[node]).unwrap();
        let tape = shape.interval_tape(Default::default());
        assert_eq!(tape.vars().len(), 1);

        for &a in args.iter() {
            let (o, trace) = eval.eval(&tape, &[a]).unwrap();
            let o = o[0];
            assert!(trace.is_none());

            for i in 0..32 {
                let pos = i as f32 / 31.0;
                let inside = (a.lower() * pos + a.upper() * (1.0 - pos))
                    .min(a.upper())
                    .max(a.lower());
                let inside_value = C::eval_f32(inside);

                if inside_value.is_nan() || inside_value.is_infinite() {
                    assert!(
                        o.has_nan(),
                        "interval failure in '{}': {inside} in {a} => \
                             {inside_value} not in {o} (should be [NaN, NaN])",
                        C::NAME,
                    );
                } else if !o.has_nan() {
                    assert!(
                        inside_value >= o.lower() && inside_value <= o.upper(),
                        "interval failure in '{}': {inside} in {a} => \
                             {inside_value} not in {o}",
                        C::NAME,
                    );
                }
            }
        }
    }

    /// Check `out` against a grid of points in the LHS, RHS intervals
    pub fn compare_interval_results(
        lhs: Interval,
        rhs: Interval,
        out: Interval,
        g: impl Fn(f32, f32) -> f32,
        name: &str,
    ) {
        let i_max = if lhs.lower() == lhs.upper() { 1 } else { 8 };
        let j_max = if rhs.lower() == rhs.upper() { 1 } else { 8 };
        for i in 0..i_max {
            for j in 0..j_max {
                let i = i as f32 / (i_max - 1) as f32;
                let j = j as f32 / (j_max - 1) as f32;
                let v_lhs = (lhs.lower() * i + lhs.upper() * (1.0 - i))
                    .min(lhs.upper())
                    .max(lhs.lower());
                let v_rhs = (rhs.lower() * j + rhs.upper() * (1.0 - j))
                    .min(rhs.upper())
                    .max(rhs.lower());
                let inside_value = g(v_lhs, v_rhs);

                if inside_value.is_nan() || inside_value.is_infinite() {
                    assert!(
                        out.has_nan(),
                        "interval failure in '{name}': ({v_lhs}, {v_rhs}) in \
                        ({lhs}, {rhs}) => {inside_value} not in {out} \
                        (should be [NaN, NaN])"
                    );
                } else if !out.has_nan() {
                    assert!(
                        inside_value >= out.lower()
                            && inside_value <= out.upper(),
                        "interval failure in '{name}': ({v_lhs}, {v_rhs}) in \
                        ({lhs}, {rhs}) => {inside_value} not in {out}"
                    );
                }
            }
        }
    }

    pub fn test_binary_reg_reg<C: CanonicalBinaryOp>() {
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let va = Var::new();
        let vb = Var::new();
        let a = ctx.var(va);
        let b = ctx.var(vb);

        let name = format!("{}(reg, reg)", C::NAME);
        let mut tape_data = None;
        let mut eval = F::new_interval_eval();
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                let node = C::build(&mut ctx, a, b);

                // Special-case for things like x * x, which get
                // optimized to x**2 (with a different interval result)
                let op = ctx.get_op(node).unwrap();
                if matches!(op, crate::context::Op::Unary(..)) {
                    continue;
                }

                let shape = F::new(&ctx, &[node]).unwrap();
                let tape = shape.interval_tape(tape_data.unwrap_or_default());

                let vars = tape.vars();

                let i_index = vars[&va];
                let j_index = vars[&vb];
                assert_ne!(i_index, j_index);

                let mut args = [0f32.into(); 2];
                args[j_index] = rhs;
                args[i_index] = lhs;

                let (out, _trace) = eval.eval(&tape, &args).unwrap();
                tape_data = Some(tape.recycle().unwrap());

                Self::compare_interval_results(
                    lhs,
                    rhs,
                    out[0],
                    C::eval_reg_reg_f32,
                    &name,
                );
            }
        }

        for &lhs in args.iter() {
            let node = C::build(&mut ctx, a, a);

            // Special-case for things like x * x, which get
            // optimized to x**2 (with a different interval result)
            let op = ctx.get_op(node).unwrap();
            if matches!(op, crate::context::Op::Unary(..)) {
                continue;
            }

            let shape = F::new(&ctx, &[node]).unwrap();
            let tape = shape.interval_tape(tape_data.unwrap_or_default());

            let vars = tape.vars();

            let i_index = vars[&va];
            assert_eq!(i_index, 0);

            let args = [lhs];
            let (out, _trace) = eval.eval(&tape, &args).unwrap();
            tape_data = Some(tape.recycle().unwrap());

            Self::compare_interval_results(
                lhs,
                lhs,
                out[0],
                C::eval_reg_reg_f32,
                &name,
            );
        }
    }

    pub fn test_binary_reg_imm<C: CanonicalBinaryOp>() {
        let values = test_args();
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let va = Var::new();
        let a = ctx.var(va);

        let name = format!("{}(reg, imm)", C::NAME);
        let mut tape_data = None;
        let mut eval = F::new_interval_eval();
        for &lhs in args.iter() {
            for &rhs in values.iter() {
                let node = C::build(&mut ctx, a, rhs);

                let shape = F::new(&ctx, &[node]).unwrap();
                let tape = shape.interval_tape(tape_data.unwrap_or_default());

                let (out, _trace) = eval.eval(&tape, &[lhs]).unwrap();
                tape_data = Some(tape.recycle().unwrap());

                Self::compare_interval_results(
                    lhs,
                    rhs.into(),
                    out[0],
                    C::eval_reg_imm_f32,
                    &name,
                );
            }
        }
    }

    pub fn test_binary_imm_reg<C: CanonicalBinaryOp>() {
        let values = test_args();
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let va = Var::new();

        let name = format!("{}(imm, reg)", C::NAME);
        let mut tape_data = None;
        let mut eval = F::new_interval_eval();
        for &lhs in values.iter() {
            for &rhs in args.iter() {
                let node = C::build(&mut ctx, lhs, va);

                let shape = F::new(&ctx, &[node]).unwrap();
                let tape = shape.interval_tape(tape_data.unwrap_or_default());

                let (out, _trace) = eval.eval(&tape, &[rhs]).unwrap();
                tape_data = Some(tape.recycle().unwrap());

                Self::compare_interval_results(
                    lhs.into(),
                    rhs,
                    out[0],
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
macro_rules! interval_test {
    ($i:ident, $t:ty) => {
        #[test]
        fn $i() {
            $crate::eval::test::interval::TestInterval::<$t>::$i()
        }
    };
}

#[macro_export]
macro_rules! interval_tests {
    ($t:ty) => {
        $crate::interval_test!(test_interval, $t);
        $crate::interval_test!(test_i_abs, $t);
        $crate::interval_test!(test_i_add_abs, $t);
        $crate::interval_test!(test_i_sqrt, $t);
        $crate::interval_test!(test_i_square, $t);
        $crate::interval_test!(test_i_sin, $t);
        $crate::interval_test!(test_i_neg, $t);
        $crate::interval_test!(test_i_not, $t);
        $crate::interval_test!(test_i_mul, $t);
        $crate::interval_test!(test_i_mul_imm, $t);
        $crate::interval_test!(test_i_sub, $t);
        $crate::interval_test!(test_i_sub_imm, $t);
        $crate::interval_test!(test_i_recip, $t);
        $crate::interval_test!(test_i_div, $t);
        $crate::interval_test!(test_i_min, $t);
        $crate::interval_test!(test_i_min_imm, $t);
        $crate::interval_test!(test_i_max, $t);
        $crate::interval_test!(test_i_max_imm, $t);
        $crate::interval_test!(test_i_and, $t);
        $crate::interval_test!(test_i_or, $t);
        $crate::interval_test!(test_i_compare, $t);
        $crate::interval_test!(test_i_simplify, $t);
        $crate::interval_test!(test_i_simplify_conditional, $t);
        $crate::interval_test!(test_i_stress, $t);

        mod i_unary {
            use super::*;
            $crate::all_unary_tests!(
                $crate::eval::test::interval::TestInterval::<$t>
            );
        }

        mod i_binary {
            use super::*;
            $crate::all_binary_tests!(
                $crate::eval::test::interval::TestInterval::<$t>
            );
        }
    };
}
