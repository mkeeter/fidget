//! Single-point evaluation
use crate::eval::{
    tracing::{TracingEval, TracingEvalData, TracingEvaluator},
    EvaluatorStorage, Family,
};

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for a single point, returning an `f32` and capturing a trace
pub type PointEval<F> = TracingEval<f32, <F as Family>::PointEval, F>;

/// Scratch data used by an point evaluator from a particular family `F`
pub type PointEvalData<F> = TracingEvalData<
    <<F as Family>::PointEval as TracingEvaluator<f32, F>>::Data,
    F,
>;

/// Immutable data used by an interval evaluator from a particular family `F`
pub type PointEvalStorage<F> =
    <<F as Family>::PointEval as EvaluatorStorage<F>>::Storage;

////////////////////////////////////////////////////////////////////////////////

// This module exports a standard test suite for any point evaluator, which can
// be included as `point_tests!(ty)`.
#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{Choice, Vars},
    };

    pub fn test_constant<I: Family>() {
        let mut ctx = Context::new();
        let p = ctx.constant(1.5);
        let tape = ctx.get_tape::<I>(p).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(0.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_constant_push<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.constant(1.5);
        let x = ctx.x();
        let min = ctx.min(a, x).unwrap();
        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_point_evaluator();
        let (r, data) = eval.eval(2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.5);

        let next = data.unwrap().simplify().unwrap();
        assert_eq!(next.len(), 1);

        let eval = next.new_point_evaluator();
        assert_eq!(eval.eval(2.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
        assert_eq!(eval.eval(1.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_circle<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let circle = ctx.sub(radius, 1.0).unwrap();

        let tape = ctx.get_tape::<I>(circle).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(0.0, 0.0, 0.0, &[]).unwrap().0, -1.0);
        assert_eq!(eval.eval(1.0, 0.0, 0.0, &[]).unwrap().0, 0.0);
    }

    pub fn test_p_min<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_point_evaluator();
        let (r, data) = eval.eval(0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(data.is_none());

        let (r, data) = eval.eval(0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (r, data) = eval.eval(2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (r, data) = eval.eval(std::f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());

        let (r, data) = eval.eval(0.0, std::f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());
    }

    pub fn test_p_max<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape::<I>(max).unwrap();
        let eval = tape.new_point_evaluator();

        let (r, data) = eval.eval(0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(data.is_none());

        let (r, data) = eval.eval(0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (r, data) = eval.eval(2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 2.0);
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (r, data) = eval.eval(std::f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());

        let (r, data) = eval.eval(0.0, std::f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());
    }

    pub fn basic_interpreter<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sum = ctx.add(x, 1.0).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(1.0, 3.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(3.0, 3.5, 0.0, &[]).unwrap().0, 3.5);
    }

    pub fn test_push<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.clone().new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let t = tape.simplify(&[Choice::Left]).unwrap();
        let eval = t.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(3.0, 2.0, 0.0, &[]).unwrap().0, 3.0);

        let t = tape.simplify(&[Choice::Right]).unwrap();
        let eval = t.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let min = ctx.min(x, 1.0).unwrap();
        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.clone().new_point_evaluator();
        assert_eq!(eval.eval(0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);

        let t = tape.simplify(&[Choice::Left]).unwrap();
        let eval = t.new_point_evaluator();
        assert_eq!(eval.eval(0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(3.0, 0.0, 0.0, &[]).unwrap().0, 3.0);

        let t = tape.simplify(&[Choice::Right]).unwrap();
        let eval = t.new_point_evaluator();
        assert_eq!(eval.eval(0.5, 0.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);
    }

    pub fn test_basic<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape::<I>(x).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(3.0, 4.0, 0.0, &[]).unwrap().0, 3.0);

        let tape = ctx.get_tape::<I>(y).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(3.0, 4.0, 0.0, &[]).unwrap().0, 4.0);

        let y2 = ctx.mul(y, 2.5).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let tape = ctx.get_tape::<I>(sum).unwrap();
        let eval = tape.new_point_evaluator();
        assert_eq!(eval.eval(1.0, 2.0, 0.0, &[]).unwrap().0, 6.0);
    }

    pub fn test_var<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = ctx.get_tape::<I>(min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = tape.new_point_evaluator();

        assert_eq!(
            eval.eval(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 5.0), ("b", 3.0)].into_iter())
            )
            .unwrap()
            .0,
            2.0
        );
        assert_eq!(
            eval.eval(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 3.0), ("b", 2.0)].into_iter())
            )
            .unwrap()
            .0,
            2.0
        );
        assert_eq!(
            eval.eval(
                0.0,
                0.0,
                0.0,
                vars.bind([("a", 0.0), ("b", 2.0)].into_iter())
            )
            .unwrap()
            .0,
            0.5
        );
    }

    #[macro_export]
    macro_rules! point_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::point::eval_tests::$i::<$t>()
            }
        };
    }

    #[macro_export]
    macro_rules! point_tests {
        ($t:ty) => {
            $crate::point_test!(test_constant, $t);
            $crate::point_test!(test_constant_push, $t);
            $crate::point_test!(test_circle, $t);
            $crate::point_test!(test_p_max, $t);
            $crate::point_test!(test_p_min, $t);
            $crate::point_test!(basic_interpreter, $t);
            $crate::point_test!(test_push, $t);
            $crate::point_test!(test_var, $t);
            $crate::point_test!(test_basic, $t);
        };
    }
}
