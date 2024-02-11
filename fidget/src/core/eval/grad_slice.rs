//! Test suite for partial derivative (gradient) evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use crate::{
        context::Context,
        eval::{
            types::Grad, BulkEvaluator, MathShape, ShapeGradSliceEval, Vars,
        },
    };

    pub fn test_g_x<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let tape = S::new(&ctx, x).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_y<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let y = ctx.y();
        let tape = S::new(&ctx, y).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_z<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let z = ctx.z();
        let tape = S::new(&ctx, z).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(4.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_square<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.square(x).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[0.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 4.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[3.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(9.0, 6.0, 0.0, 0.0)
        );
    }

    pub fn test_g_abs<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.abs(x).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[-2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, -1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_sqrt<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sqrt(x).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 0.5, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_mul<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let s = ctx.mul(x, y).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 4.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(8.0, 2.0, 4.0, 0.0)
        );
    }

    pub fn test_g_div<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.div(x, 2.0).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.5, 0.0, 0.0)
        );
    }

    pub fn test_g_recip<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.recip(x).unwrap();
        let tape = S::new(&ctx, s).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, -1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, -0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_min<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.min(x, y).unwrap();
        let tape = S::new(&ctx, m).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_min_max<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();
        let min = ctx.min(x, y).unwrap();
        let max = ctx.max(min, z).unwrap();
        let tape = S::new(&ctx, max).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[3.0], &[5.0], &[]).unwrap()[0],
            Grad::new(5.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_max<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.max(x, y).unwrap();
        let tape = S::new(&ctx, m).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_circle<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let sqrt = ctx.sqrt(sum).unwrap();
        let sub = ctx.sub(sqrt, 0.5).unwrap();
        let tape = S::new(&ctx, sub).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&t, &[0.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_var<S: ShapeGradSliceEval + MathShape>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let tape = S::new(&ctx, a).unwrap();
        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&t, &[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            2.0.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let div = ctx.div(sum, 2.0).unwrap();
        let tape = S::new(&ctx, div).unwrap();

        let eval = S::Eval::new();
        let t = tape.tape();
        assert_eq!(
            eval.eval(&t, &[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&t, &[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            1.5.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = S::new(&ctx, min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = S::Eval::new();
        let t = tape.tape();

        assert_eq!(
            eval.eval(
                &t,
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
                &t,
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
                &t,
                &[0.0],
                &[0.0],
                &[0.0],
                vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
            )
            .unwrap()[0],
            0.5.into(),
        );
    }

    #[macro_export]
    macro_rules! grad_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::grad_slice::eval_tests::$i::<$t>()
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
            $crate::grad_test!(test_g_mul, $t);
            $crate::grad_test!(test_g_min, $t);
            $crate::grad_test!(test_g_max, $t);
            $crate::grad_test!(test_g_min_max, $t);
            $crate::grad_test!(test_g_div, $t);
            $crate::grad_test!(test_g_recip, $t);
            $crate::grad_test!(test_g_var, $t);
        };
    }
}
