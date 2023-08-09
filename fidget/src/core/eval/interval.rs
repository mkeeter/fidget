//! Interval evaluation
use crate::eval::{
    tracing::{TracingEval, TracingEvalData, TracingEvaluator},
    types::Interval,
    EvaluatorStorage, Family,
};

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for intervals, returning an interval and capturing a trace
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
    use crate::eval::types::Choice;

    #[test]
    fn test_interval() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(0.5, 1.5);
        let (v, c) = a.min_choice(b);
        assert_eq!(v, [0.0, 1.0].into());
        assert_eq!(c, Choice::BothValues);
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(any(test, feature = "eval-tests"))]
pub mod eval_tests {
    use super::*;
    use crate::{context::Context, eval::Vars};

    pub fn test_interval<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape::<I>(x).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_xy([0.0, 1.0], [2.0, 3.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [2.0, 3.0]), [1.0, 5.0].into());

        let tape = ctx.get_tape::<I>(y).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_xy([0.0, 1.0], [2.0, 3.0]), [2.0, 3.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [4.0, 5.0]), [4.0, 5.0].into());
    }

    pub fn test_i_abs<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let tape = ctx.get_tape::<I>(abs_x).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x([1.0, 5.0]), [1.0, 5.0].into());
        assert_eq!(eval.eval_x([-2.0, 5.0]), [0.0, 5.0].into());
        assert_eq!(eval.eval_x([-6.0, 5.0]), [0.0, 6.0].into());
        assert_eq!(eval.eval_x([-6.0, -1.0]), [1.0, 6.0].into());

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let tape = ctx.get_tape::<I>(sum).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [-2.0, 3.0]), [1.0, 8.0].into());
        assert_eq!(eval.eval_xy([1.0, 5.0], [-4.0, 3.0]), [1.0, 9.0].into());
    }

    pub fn test_i_sqrt<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let tape = ctx.get_tape::<I>(sqrt_x).unwrap();
        let eval = tape.new_interval_evaluator();
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

        let tape = ctx.get_tape::<I>(sqrt_x).unwrap();
        let eval = tape.new_interval_evaluator();
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

    pub fn test_i_neg<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let neg_x = ctx.neg(x).unwrap();

        let tape = ctx.get_tape::<I>(neg_x).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [-1.0, 0.0].into());
        assert_eq!(eval.eval_x([0.0, 4.0]), [-4.0, 0.0].into());
        assert_eq!(eval.eval_x([2.0, 4.0]), [-4.0, -2.0].into());
        assert_eq!(eval.eval_x([-2.0, 4.0]), [-4.0, 2.0].into());
        assert_eq!(eval.eval_x([-6.0, -2.0]), [2.0, 6.0].into());
        assert_eq!(eval.eval_x([-6.0, 1.0]), [-1.0, 6.0].into());

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

        let tape = ctx.get_tape::<I>(mul).unwrap();
        let eval = tape.new_interval_evaluator();
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
        let tape = ctx.get_tape::<I>(mul).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [2.0, 4.0].into());

        let mul = ctx.mul(x, -3.0).unwrap();
        let tape = ctx.get_tape::<I>(mul).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [-3.0, 0.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-6.0, -3.0].into());
    }

    pub fn test_i_sub<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let tape = ctx.get_tape::<I>(sub).unwrap();
        let eval = tape.new_interval_evaluator();
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
        let tape = ctx.get_tape::<I>(sub).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [-2.0, -1.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-1.0, 0.0].into());

        let sub = ctx.sub(-3.0, x).unwrap();
        let tape = ctx.get_tape::<I>(sub).unwrap();
        let eval = tape.new_interval_evaluator();
        assert_eq!(eval.eval_x([0.0, 1.0]), [-4.0, -3.0].into());
        assert_eq!(eval.eval_x([1.0, 2.0]), [-5.0, -4.0].into());
    }

    pub fn test_i_recip<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let tape = ctx.get_tape::<I>(recip).unwrap();
        let eval = tape.new_interval_evaluator();

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
        let tape = ctx.get_tape::<I>(div).unwrap();
        let eval = tape.new_interval_evaluator();

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

        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_interval_evaluator();
        let (r, data) =
            eval.eval([0.0, 1.0], [0.5, 1.5], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([0.0, 1.0], [2.0, 3.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);

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

        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_interval_evaluator();
        let (r, data) = eval.eval([0.0, 1.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([-1.0, 0.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [-1.0, 0.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);

        let (r, data) = eval.eval([2.0, 3.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);
    }

    pub fn test_i_max<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape::<I>(max).unwrap();
        let eval = tape.new_interval_evaluator();
        let (r, data) =
            eval.eval([0.0, 1.0], [0.5, 1.5], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.5, 1.5].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([0.0, 1.0], [2.0, 3.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);

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
        let tape = ctx.get_tape::<I>(max_xy_z).unwrap();
        let eval = tape.new_interval_evaluator();

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [4.0, 5.0], &[]).unwrap();
        assert_eq!(r, [4.0, 5.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (3 * 2))]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [1.0, 4.0], &[]).unwrap();
        assert_eq!(r, [2.0, 4.0].into());
        assert_eq!(
            data.unwrap().choices(),
            &[3 | (1 << (1 * 2)) | (1 << (3 * 2))]
        );

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 1.0], [1.0, 1.5], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);
    }

    pub fn test_i_simplify<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let tape = ctx.get_tape::<I>(min).unwrap();
        let eval = tape.new_interval_evaluator();
        let (out, data) =
            eval.eval([0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 1.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval([0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 0.5].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);

        let (out, data) =
            eval.eval([1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);

        let max = ctx.max(x, 1.0).unwrap();
        let tape = ctx.get_tape::<I>(max).unwrap();
        let eval = tape.new_interval_evaluator();
        let (out, data) =
            eval.eval([0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 2.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval([0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);

        let (out, data) =
            eval.eval([1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.5, 2.5].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);
    }

    pub fn test_i_max_imm<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let max = ctx.max(x, 1.0).unwrap();

        let tape = ctx.get_tape::<I>(max).unwrap();
        let eval = tape.new_interval_evaluator();
        let (r, data) =
            eval.eval([0.0, 2.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [1.0, 2.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (2 * 2))]);

        let (r, data) =
            eval.eval([2.0, 3.0], [0.0, 0.0], [0.0, 0.0], &[]).unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().choices(), &[3 | (1 << (1 * 2))]);
    }

    pub fn test_i_var<I: Family>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
        let b = ctx.var("b").unwrap();
        let sum = ctx.add(a, 1.0).unwrap();
        let min = ctx.div(sum, b).unwrap();
        let tape = ctx.get_tape::<I>(min).unwrap();
        let mut vars = Vars::new(&tape);
        let eval = tape.new_interval_evaluator();

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

    pub fn test_i_fancy_simplify<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let corner = ctx.max(x, y).unwrap();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let r = ctx.sqrt(sum).unwrap();
        let circle = ctx.sub(r, 0.5).unwrap();
        let out = ctx.min(corner, circle).unwrap();

        let tape = ctx.get_tape::<I>(out).unwrap();
        let initial_size = tape.len();

        // Rough visualization of our model:
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                                                 |
        //                                 XX                              |
        //                           XXXXXXXXXXXXXX                        |
        //                       XXXXXXXXXXXXXXXXXXXXXX                    |
        //                     XXXXXXXXXXXXXXXXXXXXXXXXXX                  |
        //                     XXXXXXXXXXXXXXXXXXXXXXXXXX                  |
        //                   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        //                   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        //                   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                  |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                  |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                    |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                        |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                |
        //
        // We're going to perform interval evaluation on the upper-right corner,
        // which should flatten out to the circle.

        let eval = tape.new_interval_evaluator();
        let (_out, data) =
            eval.eval([0.6, 0.7], [0.6, 0.7], [0.0; 2], &[]).unwrap();
        let tape = data.unwrap().simplify();

        // We expect to prune the following:
        // - MinRegRegChoice (to combine the corner with the circle)
        // - MaxRegRegChoice (for x side of corner)
        // - MaxRegRegChoice (for y side of corner)
        assert_eq!(tape.len(), initial_size - 3);
    }

    pub fn test_i_rect_simplify<I: Family>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let a = ctx.sub(y, 0.55).unwrap();
        let ny = ctx.neg(y).unwrap();
        let m1 = ctx.max(a, ny).unwrap();
        let b = ctx.sub(x, 0.825).unwrap();
        let m2 = ctx.max(m1, b).unwrap();
        let c = ctx.sub(0.725, x).unwrap();
        let out = ctx.max(m2, c).unwrap();

        let tape = ctx.get_tape::<I>(out).unwrap();
        let eval = tape.new_interval_evaluator();

        // Pick a region that should only be affected by component '0.725 - X'
        let r = eval
            .eval([0.5, 0.75], [0.25, 0.5], [0.0, 0.0], &[])
            .unwrap();

        let shorter = r.1.unwrap().simplify();
        let shorter_eval = shorter.new_interval_evaluator();

        let a = eval
            .eval([0.5, 0.75], [0.25, 0.5], [0.0, 0.0], &[])
            .unwrap();
        let b = shorter_eval
            .eval([0.5, 0.75], [0.25, 0.5], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(a.0, b.0);
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
            $crate::interval_test!(test_i_fancy_simplify, $t);
            $crate::interval_test!(test_i_rect_simplify, $t);
            $crate::interval_test!(test_i_var, $t);
        };
    }
}
