//! Evaluation of partial derivatives
use crate::eval::{
    bulk::{BulkEval, BulkEvalData, BulkEvaluator},
    types::Grad,
    EvaluatorStorage, Family,
};

/// Evaluator for many points, calculating partial derivatives
pub type GradSliceEval<F> = BulkEval<Grad, <F as Family>::GradSliceEval, F>;

/// Scratch data used by an bulk gradient evaluator from a particular family `F`
pub type GradSliceEvalData<F> = BulkEvalData<
    <<F as Family>::GradSliceEval as BulkEvaluator<Grad, F>>::Data,
    Grad,
    F,
>;

/// Immutable data used by an bulk gradient evaluator from a particular family
/// `F`
pub type GradSliceEvalStorage<F> =
    <<F as Family>::GradSliceEval as EvaluatorStorage<F>>::Storage;

////////////////////////////////////////////////////////////////////////////////

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{Tape, Vars},
    };

    pub fn test_g_x<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let tape = Tape::<I>::new(&ctx, x).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_y<I: Family>() {
        let mut ctx = Context::new();
        let y = ctx.y();
        let tape = Tape::<I>::new(&ctx, y).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_z<I: Family>() {
        let mut ctx = Context::new();
        let z = ctx.z();
        let tape = Tape::<I>::new(&ctx, z).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[4.0], &[]).unwrap()[0],
            Grad::new(4.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_square<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.square(x).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[0.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 4.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[3.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(9.0, 6.0, 0.0, 0.0)
        );
    }

    pub fn test_g_abs<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.abs(x).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[-2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, -1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_sqrt<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sqrt(x).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, 0.5, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_mul<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let s = ctx.mul(x, y).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 4.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(8.0, 2.0, 4.0, 0.0)
        );
    }

    pub fn test_g_div<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.div(x, 2.0).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.5, 0.0, 0.0)
        );
    }

    pub fn test_g_recip<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.recip(x).unwrap();
        let tape = Tape::<I>::new(&ctx, s).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.0, -1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, -0.25, 0.0, 0.0)
        );
    }

    pub fn test_g_min<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.min(x, y).unwrap();
        let tape = Tape::<I>::new(&ctx, m).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_min_max<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();
        let min = ctx.min(x, y).unwrap();
        let max = ctx.max(min, z).unwrap();
        let tape = Tape::<I>::new(&ctx, max).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(2.0, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[3.0], &[5.0], &[]).unwrap()[0],
            Grad::new(5.0, 0.0, 0.0, 1.0)
        );
    }

    pub fn test_g_max<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let m = ctx.max(x, y).unwrap();
        let tape = Tape::<I>::new(&ctx, m).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[2.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(3.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[4.0], &[3.0], &[0.0], &[]).unwrap()[0],
            Grad::new(4.0, 1.0, 0.0, 0.0)
        );
    }

    pub fn test_g_circle<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let sqrt = ctx.sqrt(sum).unwrap();
        let sub = ctx.sub(sqrt, 0.5).unwrap();
        let tape = Tape::<I>::new(&ctx, sub).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[1.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[0.0], &[1.0], &[0.0], &[]).unwrap()[0],
            Grad::new(0.5, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[2.0], &[0.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 1.0, 0.0, 0.0)
        );
        assert_eq!(
            eval.eval(&[0.0], &[2.0], &[0.0], &[]).unwrap()[0],
            Grad::new(1.5, 0.0, 1.0, 0.0)
        );
    }

    pub fn test_g_var<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let tape = Tape::<I>::new(&ctx, a).unwrap();
        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            2.0.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let div = ctx.div(sum, 2.0).unwrap();
        let tape = Tape::<I>::new(&ctx, div).unwrap();

        let eval = tape.new_grad_slice_evaluator();
        assert_eq!(
            eval.eval(&[0.0], &[0.0], &[0.0], &[1.0]).unwrap()[0],
            1.0.into()
        );
        assert_eq!(
            eval.eval(&[0.0], &[0.0], &[0.0], &[2.0]).unwrap()[0],
            1.5.into()
        );

        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = Tape::<I>::new(&ctx, min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = tape.new_grad_slice_evaluator();

        assert_eq!(
            eval.eval(
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
