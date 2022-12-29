//! Interval evaluation
use crate::eval::{
    tracing::{TracingEval, TracingEvalData, TracingEvaluator},
    types::Interval,
    EvaluatorStorage, Family,
};

////////////////////////////////////////////////////////////////////////////////

/// User-friendly interval evaluator
pub type IntervalEval<F> =
    TracingEval<Interval, <F as Family>::IntervalEval, F>;

/// Scratch data used by an interval evaluator from a particular family `F`
pub type IntervalEvalData<F> = TracingEvalData<
    <<F as Family>::IntervalEval as TracingEvaluator<Interval, F>>::Data,
    F,
>;

/// Immutable data used by an interval evaluator from a particular family `F`
pub type IntervalEvalStorage<F> =
    <<F as Family>::IntervalEval as EvaluatorStorage<F>>::Storage;

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::eval::Choice;

    #[test]
    fn test_interval() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(0.5, 1.5);
        let (v, c) = a.min_choice(b);
        assert_eq!(v, [0.0, 1.0].into());
        assert_eq!(c, Choice::Both);
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{Choice, Eval, Vars},
    };

    pub fn test_interval<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape(x).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_xy([0.0, 1.0], [2.0, 3.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [2.0, 3.0]), [1.0, 5.0].into());

        let tape = ctx.get_tape(y).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_xy([0.0, 1.0], [2.0, 3.0]), [2.0, 3.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [4.0, 5.0]), [4.0, 5.0].into());
    }

    pub fn test_i_abs<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let tape = ctx.get_tape(abs_x).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x([1.0, 5.0]), [1.0, 5.0].into());
        assert_eq!(eval.eval_x([-2.0, 5.0]), [0.0, 5.0].into());
        assert_eq!(eval.eval_x([-6.0, 5.0]), [0.0, 6.0].into());
        assert_eq!(eval.eval_x([-6.0, -1.0]), [1.0, 6.0].into());

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let tape = ctx.get_tape(sum).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [-2.0, 3.0]), [1.0, 8.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [-4.0, 3.0]), [1.0, 9.0].into());
    }

    pub fn test_i_sqrt<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let tape = ctx.get_tape(sqrt_x).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x([0.0, 4.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x([-2.0, 4.0]), [0.0, 2.0].into());
        let nanan = eval.eval_x([-2.0, -1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let (v, _) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_square<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let tape = ctx.get_tape(sqrt_x).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x([0.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x([2.0, 4.0]), [4.0, 16.0].into());
        assert_eq!(eval.eval_x([-2.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x([-6.0, -2.0]), [4.0, 36.0].into());
        assert_eq!(eval.eval_x([-6.0, 1.0]), [0.0, 36.0].into());

        let (v, _) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let tape = ctx.get_tape(mul).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 2.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_xy([-2.0, 1.0], [0.0, 1.0]), [-2.0, 1.0].into());
        assert_eq!(
            eval.eval_xy([-2.0, -1.0], [-5.0, -4.0]),
            [4.0, 10.0].into()
        );
        assert_eq!(
            eval.eval_xy([-3.0, -1.0], [-2.0, 6.0]),
            [-18.0, 6.0].into()
        );

        let (v, _) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul_imm<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let mul = ctx.mul(x, 2.0).unwrap();
        let tape = ctx.get_tape(mul).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [2.0, 4.0].into());

        let mul = ctx.mul(x, -3.0).unwrap();
        let tape = ctx.get_tape(mul).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [-3.0, 0.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-6.0, -3.0].into());
    }

    pub fn test_i_sub<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let tape = ctx.get_tape(sub).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 1.0]), [-1.0, 1.0].into());
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 2.0]), [-2.0, 1.0].into());
        assert_eq!(eval.eval_xy([-2.0, 1.0], [0.0, 1.0]), [-3.0, 1.0].into());
        assert_eq!(eval.eval_xy([-2.0, -1.0], [-5.0, -4.0]), [2.0, 4.0].into());
        assert_eq!(eval.eval_xy([-3.0, -1.0], [-2.0, 6.0]), [-9.0, 1.0].into());
    }

    pub fn test_i_sub_imm<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sub = ctx.sub(x, 2.0).unwrap();
        let tape = ctx.get_tape(sub).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [-2.0, -1.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-1.0, 0.0].into());

        let sub = ctx.sub(-3.0, x).unwrap();
        let tape = ctx.get_tape(sub).unwrap();
        let eval = I::new_interval_evaluator(tape);
        assert_eq!(eval.eval_x([0.0, 1.0]), [-4.0, -3.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-5.0, -4.0].into());
    }

    pub fn test_i_recip<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let tape = ctx.get_tape(recip).unwrap();
        let eval = I::new_interval_evaluator(tape);

        let nanan = eval.eval_x([0.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x([-1.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x([-2.0, 3.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        assert_eq!(eval.eval_x([-2.0, -1.0]), [-1.0, -0.5].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [0.5, 1.0].into());
    }

    pub fn test_i_div<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let div = ctx.div(x, y).unwrap();
        let tape = ctx.get_tape(div).unwrap();
        let eval = I::new_interval_evaluator(tape);

        let nanan = eval.eval_xy([0.0, 1.0], [-1.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy([0.0, 1.0], [-2.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy([0.0, 1.0], [0.0, 4.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let out = eval.eval_xy([-1.0, 0.0], [1.0, 2.0]);
        assert_eq!(out, [-1.0, 0.0].into());

        let out = eval.eval_xy([-1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, 2.0].into());

        let out = eval.eval_xy([1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, -1.0].into());

        let out = eval.eval_xy([-1.0, 4.0], [0.5, 1.0]);
        assert_eq!(out, [-2.0, 8.0].into());

        let (v, _) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_min<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (r, data) =
            eval.eval([0.0, 1.0], [0.5, 1.5], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([0.0, 1.0], [2.0, 3.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (v, data) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());
    }

    pub fn test_i_min_imm<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let tape = ctx.get_tape(min).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (r, data) = eval.eval([0.0, 1.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([-1.0, 0.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [-1.0, 0.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (r, data) = eval.eval([2.0, 3.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);
    }

    pub fn test_i_max<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape(max).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (r, data) =
            eval.eval([0.0, 1.0], [0.5, 1.5], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.5, 1.5].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([0.0, 1.0], [2.0, 3.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (v, data) = eval
            .eval([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let tape = ctx.get_tape(max_xy_z).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [4.0, 5.0], &[]).unwrap();
        assert_eq!(r, [4.0, 5.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left, Choice::Right]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [1.0, 4.0], &[]).unwrap();
        assert_eq!(r, [2.0, 4.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left, Choice::Both]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [1.0, 1.5], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left, Choice::Left]);
    }

    pub fn test_i_simplify<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let tape = ctx.get_tape(min).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (out, data) =
            eval.eval([0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 1.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval([0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 0.5].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);

        let (out, data) =
            eval.eval([1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let max = ctx.max(x, 1.0).unwrap();
        let tape = ctx.get_tape(max).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (out, data) =
            eval.eval([0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 2.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval([0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (out, data) =
            eval.eval([1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.5, 2.5].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);
    }

    pub fn test_i_max_imm<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let max = ctx.max(x, 1.0).unwrap();

        let tape = ctx.get_tape(max).unwrap();
        let eval = I::new_interval_evaluator(tape);
        let (r, data) =
            eval.eval([0.0, 2.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [1.0, 2.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Right]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[Choice::Left]);
    }

    pub fn test_i_var<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = ctx.get_tape(min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = I::new_interval_evaluator(tape);

        assert_eq!(
            eval.eval(
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

    #[macro_export]
    macro_rules! interval_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::interval::eval_tests::$i::<$t>()
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
        };
    }
}
