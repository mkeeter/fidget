//! Test suite for single-point tracing evaluation
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for point evaluators; otherwise, the module has no public exports.
#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use crate::{
        context::{Context, Node},
        eval::{Choice, Shape, ShapeVars, TracingEvaluator, Vars},
    };

    /// Helper struct to put constrains on our `Shape` object
    pub struct TestPoint<S>(std::marker::PhantomData<*const S>);
    impl<S> TestPoint<S>
    where
        for<'a> S: Shape + TryFrom<(&'a Context, Node)> + ShapeVars,
        for<'a> <S as TryFrom<(&'a Context, Node)>>::Error: std::fmt::Debug,
        <S as Shape>::Trace: AsRef<[Choice]>,
        <S as Shape>::Trace: From<Vec<Choice>>,
    {
        pub fn test_constant() {
            let mut ctx = Context::new();
            let p = ctx.constant(1.5);
            let shape = S::try_from((&ctx, p)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
        }

        pub fn test_constant_push() {
            let mut ctx = Context::new();
            let a = ctx.constant(1.5);
            let x = ctx.x();
            let min = ctx.min(a, x).unwrap();
            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, 1.5);

            let next = shape.simplify(trace.unwrap(), None).unwrap();
            assert_eq!(next.size(), 1);

            let tape = next.point_tape(None);
            assert_eq!(eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
            assert_eq!(eval.eval(&tape, 1.0, 0.0, 0.0, &[]).unwrap().0, 1.5);
        }

        pub fn test_circle() {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();
            let x_squared = ctx.mul(x, x).unwrap();
            let y_squared = ctx.mul(y, y).unwrap();
            let radius = ctx.add(x_squared, y_squared).unwrap();
            let circle = ctx.sub(radius, 1.0).unwrap();

            let shape = S::try_from((&ctx, circle)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap().0, -1.0);
            assert_eq!(eval.eval(&tape, 1.0, 0.0, 0.0, &[]).unwrap().0, 0.0);
        }

        pub fn test_p_min()
        where
            <S as Shape>::Trace: AsRef<[Choice]>,
        {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();
            let min = ctx.min(x, y).unwrap();

            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            let (r, trace) = eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, 0.0);
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, 0.0, 1.0, 0.0, &[]).unwrap();
            assert_eq!(r, 0.0);
            assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

            let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, 0.0);
            assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

            let (r, trace) =
                eval.eval(&tape, std::f32::NAN, 0.0, 0.0, &[]).unwrap();
            assert!(r.is_nan());
            assert!(trace.is_none());

            let (r, trace) =
                eval.eval(&tape, 0.0, std::f32::NAN, 0.0, &[]).unwrap();
            assert!(r.is_nan());
            assert!(trace.is_none());
        }

        pub fn test_p_max()
        where
            <S as Shape>::Trace: AsRef<[Choice]>,
        {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();
            let max = ctx.max(x, y).unwrap();

            let shape = S::try_from((&ctx, max)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();

            let (r, trace) = eval.eval(&tape, 0.0, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, 0.0);
            assert!(trace.is_none());

            let (r, trace) = eval.eval(&tape, 0.0, 1.0, 0.0, &[]).unwrap();
            assert_eq!(r, 1.0);
            assert_eq!(trace.unwrap().as_ref(), &[Choice::Right]);

            let (r, trace) = eval.eval(&tape, 2.0, 0.0, 0.0, &[]).unwrap();
            assert_eq!(r, 2.0);
            assert_eq!(trace.unwrap().as_ref(), &[Choice::Left]);

            let (r, trace) =
                eval.eval(&tape, std::f32::NAN, 0.0, 0.0, &[]).unwrap();
            assert!(r.is_nan());
            assert!(trace.is_none());

            let (r, trace) =
                eval.eval(&tape, 0.0, std::f32::NAN, 0.0, &[]).unwrap();
            assert!(r.is_nan());
            assert!(trace.is_none());
        }

        pub fn basic_interpreter() {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();
            let sum = ctx.add(x, 1.0).unwrap();
            let min = ctx.min(sum, y).unwrap();
            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
            assert_eq!(eval.eval(&tape, 1.0, 3.0, 0.0, &[]).unwrap().0, 2.0);
            assert_eq!(eval.eval(&tape, 3.0, 3.5, 0.0, &[]).unwrap().0, 3.5);
        }

        pub fn test_push()
        where
            <S as Shape>::Trace: AsRef<[Choice]>,
        {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();
            let min = ctx.min(x, y).unwrap();

            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
            assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

            let next =
                shape.simplify(&vec![Choice::Left].into(), None).unwrap();
            let tape = next.point_tape(None);
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
            assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 3.0);

            let next =
                shape.simplify(&vec![Choice::Right].into(), None).unwrap();
            let tape = next.point_tape(None);
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
            assert_eq!(eval.eval(&tape, 3.0, 2.0, 0.0, &[]).unwrap().0, 2.0);

            let min = ctx.min(x, 1.0).unwrap();
            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
            assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);

            let next =
                shape.simplify(&vec![Choice::Left].into(), None).unwrap();
            let tape = next.point_tape(None);
            assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 0.5);
            assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 3.0);

            let next =
                shape.simplify(&vec![Choice::Right].into(), None).unwrap();
            let tape = next.point_tape(None);
            assert_eq!(eval.eval(&tape, 0.5, 0.0, 0.0, &[]).unwrap().0, 1.0);
            assert_eq!(eval.eval(&tape, 3.0, 0.0, 0.0, &[]).unwrap().0, 1.0);
        }

        pub fn test_basic() {
            let mut ctx = Context::new();
            let x = ctx.x();
            let y = ctx.y();

            let shape = S::try_from((&ctx, x)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 1.0);
            assert_eq!(eval.eval(&tape, 3.0, 4.0, 0.0, &[]).unwrap().0, 3.0);

            let shape = S::try_from((&ctx, y)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 2.0);
            assert_eq!(eval.eval(&tape, 3.0, 4.0, 0.0, &[]).unwrap().0, 4.0);

            let y2 = ctx.mul(y, 2.5).unwrap();
            let sum = ctx.add(x, y2).unwrap();

            let shape = S::try_from((&ctx, sum)).unwrap();
            let tape = shape.point_tape(None);
            let mut eval = S::new_point_eval();
            assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[]).unwrap().0, 6.0);
        }

        pub fn test_var() {
            let mut ctx = Context::new();
            let a = ctx.var("a").unwrap();
            let shape = S::try_from((&ctx, a)).unwrap();
            let tape = shape.point_tape(None);
            let mut vars = Vars::new(shape.vars());
            let mut eval = S::new_point_eval();
            assert_eq!(
                eval.eval(
                    &tape,
                    0.0,
                    0.0,
                    0.0,
                    vars.bind([("a", 5.0)].into_iter())
                )
                .unwrap()
                .0,
                5.0
            );
            assert_eq!(
                eval.eval(
                    &tape,
                    0.0,
                    0.0,
                    0.0,
                    vars.bind([("a", 1.0)].into_iter())
                )
                .unwrap()
                .0,
                1.0
            );

            let b = ctx.var("b").unwrap();
            let sum = ctx.add(a, 1.0).unwrap();
            let min = ctx.div(sum, b).unwrap();
            let shape = S::try_from((&ctx, min)).unwrap();
            let tape = shape.point_tape(None);
            let mut vars = Vars::new(shape.vars());
            let mut eval = S::new_point_eval();

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
                2.0
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
                2.0
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
                0.5
            );
        }
    }

    #[macro_export]
    macro_rules! point_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::point::eval_tests::TestPoint::<$t>::$i()
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
