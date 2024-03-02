//! Interval evaluation tests
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.

use super::build_stress_fn;
use crate::{
    context::Context,
    eval::{
        types::Interval, EzShape, MathShape, Shape, ShapeVars,
        TracingEvaluator, Vars,
    },
    vm::Choice,
};

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
    };
}
