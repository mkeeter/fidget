//! Test suite for single-point tracing evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for point evaluators; otherwise, the module has no public exports.
#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use crate::{
        context::Context,
        eval::{
            Choice, MathShape, ShapePointEval, ShapeVars, TracingEvaluator,
            Vars,
        },
    };

    pub fn test_constant<S: ShapePointEval + MathShape>() {
        let mut ctx = Context::new();
        let p = ctx.constant(1.5);
        let tape = S::new(&ctx, p).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 0.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_constant_push<S: ShapePointEval + MathShape>() {
        let mut ctx = Context::new();
        let a = ctx.constant(1.5);
        let x = ctx.x();
        let min = ctx.min(a, x).unwrap();
        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval.eval(&t, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.5);

        let next = tape.simplify(data.unwrap()).unwrap();
        assert_eq!(next.len(), 1);

        let t = next.tape();
        assert_eq!(eval.eval(&t, 2.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
        assert_eq!(eval.eval(&t, 1.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
    }

    pub fn test_circle<S: ShapePointEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let circle = ctx.sub(radius, 1.0).unwrap();

        let tape = S::new(&ctx, circle).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 0.0, 0.0, 0.0, &[]).unwrap().0, -1.0);
        assert_eq!(eval.eval(&t, 1.0, 0.0, 0.0, &[]).unwrap().0, 0.0);
    }

    pub fn test_p_min<S: ShapePointEval + MathShape>()
    where
        <<S as ShapePointEval>::Eval as TracingEvaluator<f32>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval.eval(&t, 0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(data.is_none());

        let (r, data) = eval.eval(&t, 0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval.eval(&t, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval.eval(&t, std::f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());

        let (r, data) = eval.eval(&t, 0.0, std::f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());
    }

    pub fn test_p_max<S: ShapePointEval + MathShape>()
    where
        <<S as ShapePointEval>::Eval as TracingEvaluator<f32>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = S::new(&ctx, max).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();

        let (r, data) = eval.eval(&t, 0.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 0.0);
        assert!(data.is_none());

        let (r, data) = eval.eval(&t, 0.0, 1.0, 0.0, &[]).unwrap();
        assert_eq!(r, 1.0);
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval.eval(&t, 2.0, 0.0, 0.0, &[]).unwrap();
        assert_eq!(r, 2.0);
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval.eval(&t, std::f32::NAN, 0.0, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());

        let (r, data) = eval.eval(&t, 0.0, std::f32::NAN, 0.0, &[]).unwrap();
        assert!(r.is_nan());
        assert!(data.is_none());
    }

    pub fn basic_interpreter<S: ShapePointEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sum = ctx.add(x, 1.0).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&t, 1.0, 3.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&t, 3.0, 3.5, 0.0, &[]).unwrap().0, 3.5);
    }

    pub fn test_push<S: ShapePointEval + MathShape>()
    where
        <<S as ShapePointEval>::Eval as TracingEvaluator<f32>>::Trace:
            From<Vec<Choice>>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&t, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let next = tape.simplify(&vec![Choice::Left].into()).unwrap();
        let t = next.tape();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&t, 3.0, 2.0, 0.0, &[]).unwrap().0, 3.0);

        let next = tape.simplify(&vec![Choice::Right].into()).unwrap();
        let t = next.tape();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&t, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

        let min = ctx.min(x, 1.0).unwrap();
        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&t, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);

        let next = tape.simplify(&vec![Choice::Left].into()).unwrap();
        let t = next.tape();
        assert_eq!(eval.eval(&t, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
        assert_eq!(eval.eval(&t, 3.0, 0.0, 0.0, &[]).unwrap().0, 3.0);

        let next = tape.simplify(&vec![Choice::Right].into()).unwrap();
        let t = next.tape();
        assert_eq!(eval.eval(&t, 0.5, 0.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&t, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);
    }

    pub fn test_basic<S: ShapePointEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = S::new(&ctx, x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
        assert_eq!(eval.eval(&t, 3.0, 4.0, 0.0, &[]).unwrap().0, 3.0);

        let tape = S::new(&ctx, y).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
        assert_eq!(eval.eval(&t, 3.0, 4.0, 0.0, &[]).unwrap().0, 4.0);

        let y2 = ctx.mul(y, 2.5).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let tape = S::new(&ctx, sum).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval(&t, 1.0, 2.0, 0.0, &[]).unwrap().0, 6.0);
    }

    pub fn test_var<S: ShapePointEval + MathShape + ShapeVars>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let tape = S::new(&ctx, a).unwrap();
        let t = tape.tape();
        let mut vars = Vars::new(tape.vars());
        let mut eval = S::Eval::new();
        assert_eq!(
            eval.eval(&t, 0.0, 0.0, 0.0, vars.bind([("a", 5.0)].into_iter()))
                .unwrap()
                .0,
            5.0
        );
        assert_eq!(
            eval.eval(&t, 0.0, 0.0, 0.0, vars.bind([("a", 1.0)].into_iter()))
                .unwrap()
                .0,
            1.0
        );

        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut vars = Vars::new(tape.vars());
        let mut eval = S::Eval::new();

        assert_eq!(
            eval.eval(
                &t,
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
                &t,
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
                &t,
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
