//! Test suite for partial derivative (gradient) evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
use crate::{
    context::Context,
    eval::{
        types::Grad, BulkEvaluator, EzShape, MathShape, Shape, ShapeVars, Vars,
    },
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
    };
}
