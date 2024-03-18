//! Interval evaluation tests
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.

use super::{build_stress_fn, test_args, test_args_n};
use crate::{
    context::{Context, Node},
    eval::{
        types::Interval, EzShape, MathShape, Shape, ShapeVars, Tape,
        TracingEvaluator, Vars,
    },
    vm::Choice,
    Error,
};

macro_rules! interval_unary {
    (Context::$i:ident, $t:expr) => {
        Self::test_unary(Context::$i, $t, stringify!($i));
    };
}

macro_rules! interval_binary {
    (Context::$i:ident, $t:expr) => {
        Self::test_binary_reg_reg(Context::$i, $t, stringify!($i));
        Self::test_binary_reg_imm(Context::$i, $t, stringify!($i));
        Self::test_binary_imm_reg(Context::$i, $t, stringify!($i));
    };
}

/// Helper struct to put constrains on our `Shape` object
pub struct TestInterval<S>(std::marker::PhantomData<*const S>);

impl<S> TestInterval<S>
where
    for<'a> S: Shape + MathShape + ShapeVars,
    <S as Shape>::Trace: AsRef<[Choice]>,
{
    pub fn test_interval() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let shape = S::new(&ctx, x).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [2.0, 3.0]),
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [1.0, 5.0], [2.0, 3.0]),
            [1.0, 5.0].into()
        );

        let shape = S::new(&ctx, y).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [2.0, 3.0]),
            [2.0, 3.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [1.0, 5.0], [4.0, 5.0]),
            [4.0, 5.0].into()
        );
    }

    pub fn test_i_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let shape = S::new(&ctx, abs_x).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 5.0]), [1.0, 5.0].into());
        assert_eq!(eval.eval_x(&tape, [-2.0, 5.0]), [0.0, 5.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, 5.0]), [0.0, 6.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, -1.0]), [1.0, 6.0].into());

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let shape = S::new(&ctx, sum).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [0.0, 1.0]),
            [0.0, 2.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [1.0, 5.0], [-2.0, 3.0]),
            [1.0, 8.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [1.0, 5.0], [-4.0, 3.0]),
            [1.0, 9.0].into()
        );
    }

    pub fn test_i_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let shape = S::new(&ctx, sqrt_x).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&tape, [0.0, 4.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x(&tape, [-2.0, 4.0]), [0.0, 2.0].into());
        let nanan = eval.eval_x(&tape, [-2.0, -1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let (v, _) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let shape = S::new(&ctx, sqrt_x).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&tape, [0.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x(&tape, [2.0, 4.0]), [4.0, 16.0].into());
        assert_eq!(eval.eval_x(&tape, [-2.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, -2.0]), [4.0, 36.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, 1.0]), [0.0, 36.0].into());

        let (v, _) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_sin() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();
        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_interval_tape();

        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [-1.0, 1.0].into());

        let y = ctx.y();
        let y = ctx.mul(y, 2.0).unwrap();
        let s = ctx.sin(y).unwrap();
        let s = ctx.add(x, s).unwrap();
        let shape = S::new(&ctx, s).unwrap();
        let tape = shape.ez_interval_tape();

        assert_eq!(eval.eval_x(&tape, [0.0, 3.0]), [-1.0, 4.0].into());
    }

    pub fn test_i_neg() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let neg_x = ctx.neg(x).unwrap();

        let shape = S::new(&ctx, neg_x).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [-1.0, 0.0].into());
        assert_eq!(eval.eval_x(&tape, [0.0, 4.0]), [-4.0, 0.0].into());
        assert_eq!(eval.eval_x(&tape, [2.0, 4.0]), [-4.0, -2.0].into());
        assert_eq!(eval.eval_x(&tape, [-2.0, 4.0]), [-4.0, 2.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, -2.0]), [2.0, 6.0].into());
        assert_eq!(eval.eval_x(&tape, [-6.0, 1.0]), [-1.0, 6.0].into());

        let (v, _) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let shape = S::new(&ctx, mul).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [0.0, 1.0]),
            [0.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [0.0, 2.0]),
            [0.0, 2.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-2.0, 1.0], [0.0, 1.0]),
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-2.0, -1.0], [-5.0, -4.0]),
            [4.0, 10.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-3.0, -1.0], [-2.0, 6.0]),
            [-18.0, 6.0].into()
        );

        let (v, _) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval(&tape, [0.0, 1.0], [f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let mul = ctx.mul(x, 2.0).unwrap();
        let shape = S::new(&ctx, mul).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 2.0]), [2.0, 4.0].into());

        let mul = ctx.mul(x, -3.0).unwrap();
        let shape = S::new(&ctx, mul).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [-3.0, 0.0].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 2.0]), [-6.0, -3.0].into());
    }

    pub fn test_i_sub() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let shape = S::new(&ctx, sub).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [0.0, 1.0]),
            [-1.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [0.0, 1.0], [0.0, 2.0]),
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-2.0, 1.0], [0.0, 1.0]),
            [-3.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-2.0, -1.0], [-5.0, -4.0]),
            [2.0, 4.0].into()
        );
        assert_eq!(
            eval.eval_xy(&tape, [-3.0, -1.0], [-2.0, 6.0]),
            [-9.0, 1.0].into()
        );
    }

    pub fn test_i_sub_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sub = ctx.sub(x, 2.0).unwrap();
        let shape = S::new(&ctx, sub).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [-2.0, -1.0].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 2.0]), [-1.0, 0.0].into());

        let sub = ctx.sub(-3.0, x).unwrap();
        let shape = S::new(&ctx, sub).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        assert_eq!(eval.eval_x(&tape, [0.0, 1.0]), [-4.0, -3.0].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 2.0]), [-5.0, -4.0].into());
    }

    pub fn test_i_recip() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let shape = S::new(&ctx, recip).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();

        let nanan = eval.eval_x(&tape, [0.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x(&tape, [-1.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x(&tape, [-2.0, 3.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        assert_eq!(eval.eval_x(&tape, [-2.0, -1.0]), [-1.0, -0.5].into());
        assert_eq!(eval.eval_x(&tape, [1.0, 2.0]), [0.5, 1.0].into());
    }

    pub fn test_i_div() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let div = ctx.div(x, y).unwrap();
        let shape = S::new(&ctx, div).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();

        let nanan = eval.eval_xy(&tape, [0.0, 1.0], [-1.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy(&tape, [0.0, 1.0], [-2.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy(&tape, [0.0, 1.0], [0.0, 4.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let out = eval.eval_xy(&tape, [-1.0, 0.0], [1.0, 2.0]);
        assert_eq!(out, [-1.0, 0.0].into());

        let out = eval.eval_xy(&tape, [-1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, 2.0].into());

        let out = eval.eval_xy(&tape, [1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, -1.0].into());

        let out = eval.eval_xy(&tape, [-1.0, 4.0], [0.5, 1.0]);
        assert_eq!(out, [-2.0, 8.0].into());

        let (v, _) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval(&tape, [0.0, 1.0], [f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, [0.0, 1.0], [0.5, 1.5], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&tape, [0.0, 1.0], [2.0, 3.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (v, data) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval(&tape, [0.0, 1.0], [f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());
    }

    pub fn test_i_min_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, [0.0, 1.0], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&tape, [-1.0, 0.0], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [-1.0, 0.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);
    }

    pub fn test_i_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let shape = S::new(&ctx, max).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, [0.0, 1.0], [0.5, 1.5], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.5, 1.5].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&tape, [0.0, 1.0], [2.0, 3.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (v, data) = eval
            .eval(&tape, [f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval(&tape, [0.0, 1.0], [f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let shape = S::new(&ctx, max_xy_z).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 1.0], [4.0, 5.0], &[])
            .unwrap();
        assert_eq!(r, [4.0, 5.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Right]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 1.0], [1.0, 4.0], &[])
            .unwrap();
        assert_eq!(r, [2.0, 4.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Both]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 1.0], [1.0, 1.5], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Left]);
    }

    pub fn test_i_simplify() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (out, data) = eval
            .eval(&tape, [0.0, 2.0], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [0.0, 1.0].into());
        assert!(data.is_none());

        let (out, data) = eval
            .eval(&tape, [0.0, 0.5], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [0.0, 0.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (out, data) = eval
            .eval(&tape, [1.5, 2.5], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let max = ctx.max(x, 1.0).unwrap();
        let shape = S::new(&ctx, max).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (out, data) = eval
            .eval(&tape, [0.0, 2.0], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [1.0, 2.0].into());
        assert!(data.is_none());

        let (out, data) = eval
            .eval(&tape, [0.0, 0.5], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (out, data) = eval
            .eval(&tape, [1.5, 2.5], [0.0; 2], [0.0; 2], &[])
            .unwrap();
        assert_eq!(out, [1.5, 2.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_max_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let max = ctx.max(x, 1.0).unwrap();

        let shape = S::new(&ctx, max).unwrap();
        let tape = shape.ez_interval_tape();
        let mut eval = S::new_interval_eval();
        let (r, data) = eval
            .eval(&tape, [0.0, 2.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [1.0, 2.0].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&tape, [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval
            .eval(&tape, [2.0, 3.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_var() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let shape = S::new(&ctx, min).unwrap();
        let tape = shape.ez_interval_tape();
        let mut vars = Vars::new(shape.vars());
        let mut eval = S::new_interval_eval();

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
            2.0.into()
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
            2.0.into()
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
            0.5.into(),
        );
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

        let shape = S::new(&ctx, node).unwrap();
        let mut eval = S::new_interval_eval();
        let tape = shape.ez_interval_tape();

        let mut out = vec![];
        for i in 0..args.len() {
            out.push(eval.eval(&tape, x[i], y[i], z[i], &[]).unwrap().0);
        }

        // Compare against the VmShape evaluator as a baseline.  It's possible
        // that S is also a VmShape, but this comparison isn't particularly
        // expensive, so we'll do it regardless.
        use crate::vm::VmShape;
        let shape = VmShape::new(&ctx, node).unwrap();
        let mut eval = VmShape::new_interval_eval();
        let tape = shape.ez_interval_tape();

        let mut cmp = vec![];
        for i in 0..args.len() {
            cmp.push(eval.eval(&tape, x[i], y[i], z[i], &[]).unwrap().0);
        }

        for (a, b) in out.iter().zip(cmp.iter()) {
            a.compare_eq(*b)
        }
    }

    pub fn test_i_stress() {
        for n in [1, 2, 4, 8, 12, 16, 32] {
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

    pub fn test_unary(
        f: impl Fn(&mut Context, Node) -> Result<Node, Error>,
        g: impl Fn(f32) -> f32,
        name: &'static str,
    ) {
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let mut tape_data = None;
        let mut eval = S::new_interval_eval();
        for (i, v) in [ctx.x(), ctx.y(), ctx.z()].into_iter().enumerate() {
            let node = f(&mut ctx, v).unwrap();

            let shape = S::new(&ctx, node).unwrap();
            let tape = shape.interval_tape(tape_data.unwrap_or_default());

            for &a in args.iter() {
                let (o, trace) = match i {
                    0 => eval.eval(&tape, a, 0.0.into(), 0.0.into(), &[]),
                    1 => eval.eval(&tape, 0.0.into(), a, 0.0.into(), &[]),
                    2 => eval.eval(&tape, 0.0.into(), 0.0.into(), a, &[]),
                    _ => unreachable!(),
                }
                .unwrap();
                assert!(trace.is_none());

                for i in 0..32 {
                    let pos = i as f32 / 31.0;
                    let inside = (a.lower() * pos + a.upper() * (1.0 - pos))
                        .min(a.upper())
                        .max(a.lower());
                    let inside_value = g(inside);
                    assert!(
                        inside_value.is_nan()
                            || o.lower().is_nan()
                            || (inside_value >= o.lower()
                                && inside_value <= o.upper()),
                        "interval failure in '{name}': {inside} in {a} => \
                         {inside_value} not in {o}"
                    );
                }
            }
            tape_data = Some(tape.recycle());
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
                assert!(
                    inside_value.is_nan()
                        || out.lower().is_nan()
                        || (inside_value >= out.lower()
                            && inside_value <= out.upper()),
                    "interval failure in '{name}': ({v_lhs}, {v_rhs}) in \
                    ({lhs}, {rhs}) => {inside_value} not in {out}"
                );
            }
        }
    }

    pub fn test_binary_reg_reg(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f32, f32) -> f32,
        name: &'static str,
    ) {
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{name}(reg, reg)");
        let zero = Interval::new(0.0, 0.0);
        let mut tape_data = None;
        let mut eval = S::new_interval_eval();
        for &lhs in args.iter() {
            for &rhs in args.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    for (j, &v) in xyz.iter().enumerate() {
                        let node = f(&mut ctx, u, v).unwrap();

                        // Special-case for things like x * x, which get
                        // optimized to x**2 (with a different interval result)
                        let op = ctx.get_op(node).unwrap();
                        if matches!(op, crate::context::Op::Unary(..)) {
                            continue;
                        }

                        let shape = S::new(&ctx, node).unwrap();
                        let tape =
                            shape.interval_tape(tape_data.unwrap_or_default());

                        let (out, _trace) = match (i, j) {
                            (0, 0) => eval.eval(&tape, lhs, zero, zero, &[]),
                            (0, 1) => eval.eval(&tape, lhs, rhs, zero, &[]),
                            (0, 2) => eval.eval(&tape, lhs, zero, rhs, &[]),
                            (1, 0) => eval.eval(&tape, rhs, lhs, zero, &[]),
                            (1, 1) => eval.eval(&tape, zero, lhs, zero, &[]),
                            (1, 2) => eval.eval(&tape, zero, lhs, rhs, &[]),
                            (2, 0) => eval.eval(&tape, rhs, zero, lhs, &[]),
                            (2, 1) => eval.eval(&tape, zero, rhs, lhs, &[]),
                            (2, 2) => eval.eval(&tape, zero, zero, lhs, &[]),
                            _ => unreachable!(),
                        }
                        .unwrap();
                        tape_data = Some(tape.recycle());

                        let rhs = if i == j { lhs } else { rhs };
                        Self::compare_interval_results(
                            lhs, rhs, out, &g, &name,
                        );
                    }
                }
            }
        }
    }

    pub fn test_binary_reg_imm(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f32, f32) -> f32,
        name: &'static str,
    ) {
        let values = test_args();
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{name}(reg, imm)");
        let zero = Interval::new(0.0, 0.0);
        let mut tape_data = None;
        let mut eval = S::new_interval_eval();
        for &lhs in args.iter() {
            for &rhs in values.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    let c = ctx.constant(rhs as f64);
                    let node = f(&mut ctx, u, c).unwrap();

                    let shape = S::new(&ctx, node).unwrap();
                    let tape =
                        shape.interval_tape(tape_data.unwrap_or_default());

                    let (out, _trace) = match i {
                        0 => eval.eval(&tape, lhs, zero, zero, &[]),
                        1 => eval.eval(&tape, zero, lhs, zero, &[]),
                        2 => eval.eval(&tape, zero, zero, lhs, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();
                    tape_data = Some(tape.recycle());

                    Self::compare_interval_results(
                        lhs,
                        rhs.into(),
                        out,
                        &g,
                        &name,
                    );
                }
            }
        }
    }

    pub fn test_binary_imm_reg(
        f: impl Fn(&mut Context, Node, Node) -> Result<Node, Error>,
        g: impl Fn(f32, f32) -> f32,
        name: &'static str,
    ) {
        let values = test_args();
        let args = Self::interval_test_args();

        let mut ctx = Context::new();
        let xyz = [ctx.x(), ctx.y(), ctx.z()];

        let name = format!("{name}(reg, imm)");
        let zero = Interval::new(0.0, 0.0);
        let mut tape_data = None;
        let mut eval = S::new_interval_eval();
        for &lhs in values.iter() {
            for &rhs in args.iter() {
                for (i, &u) in xyz.iter().enumerate() {
                    let c = ctx.constant(lhs as f64);
                    let node = f(&mut ctx, c, u).unwrap();

                    let shape = S::new(&ctx, node).unwrap();
                    let tape =
                        shape.interval_tape(tape_data.unwrap_or_default());

                    let (out, _trace) = match i {
                        0 => eval.eval(&tape, rhs, zero, zero, &[]),
                        1 => eval.eval(&tape, zero, rhs, zero, &[]),
                        2 => eval.eval(&tape, zero, zero, rhs, &[]),
                        _ => unreachable!(),
                    }
                    .unwrap();
                    tape_data = Some(tape.recycle());

                    Self::compare_interval_results(
                        lhs.into(),
                        rhs,
                        out,
                        &g,
                        &name,
                    );
                }
            }
        }
    }

    pub fn test_i_unary_ops() {
        interval_unary!(Context::neg, |v| -v);
        interval_unary!(Context::recip, |v| 1.0 / v);
        interval_unary!(Context::abs, |v| v.abs());
        interval_unary!(Context::sin, |v| v.sin());
        interval_unary!(Context::sin, |v| v.sin());
        interval_unary!(Context::cos, |v| v.cos());
        interval_unary!(Context::tan, |v| v.tan());
        interval_unary!(Context::asin, |v| v.asin());
        interval_unary!(Context::acos, |v| v.acos());
        interval_unary!(Context::atan, |v| v.atan());
        interval_unary!(Context::exp, |v| v.exp());
        interval_unary!(Context::ln, |v| v.ln());
        interval_unary!(Context::square, |v| v * v);
        interval_unary!(Context::sqrt, |v| v.sqrt());
    }

    pub fn test_i_binary_ops() {
        interval_binary!(Context::add, |a, b| a + b);
        interval_binary!(Context::sub, |a, b| a - b);

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

        interval_binary!(Context::min, |a, b| if a.is_nan() || b.is_nan() {
            f32::NAN
        } else {
            a.min(b)
        });
        interval_binary!(Context::max, |a, b| if a.is_nan() || b.is_nan() {
            f32::NAN
        } else {
            a.max(b)
        });
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
        $crate::interval_test!(test_i_sqrt, $t);
        $crate::interval_test!(test_i_square, $t);
        $crate::interval_test!(test_i_sin, $t);
        $crate::interval_test!(test_i_neg, $t);
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
        $crate::interval_test!(test_i_simplify, $t);
        $crate::interval_test!(test_i_var, $t);
        $crate::interval_test!(test_i_stress, $t);
        $crate::interval_test!(test_i_unary_ops, $t);
        $crate::interval_test!(test_i_binary_ops, $t);
    };
}
