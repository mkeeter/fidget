//! Interval evaluation tests
//!
//! If the `eval-tests` feature is set, then this exposes a standard test suite
//! for interval evaluators; otherwise, the module has no public exports.
#[cfg(test)]
mod test {
    use crate::{eval::types::Interval, eval::Choice};

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
    use crate::{
        context::Context,
        eval::{
            types::Interval, Choice, MathShape, ShapeIntervalEval, ShapeVars,
            TracingEvaluator, Vars,
        },
    };

    pub fn test_interval<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = S::new(&ctx, x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_xy(&t, [0.0, 1.0], [2.0, 3.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_xy(&t, [1.0, 5.0], [2.0, 3.0]), [1.0, 5.0].into());

        let tape = S::new(&ctx, y).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_xy(&t, [0.0, 1.0], [2.0, 3.0]), [2.0, 3.0].into());
        assert_eq!(eval.eval_xy(&t, [1.0, 5.0], [4.0, 5.0]), [4.0, 5.0].into());
    }

    pub fn test_i_abs<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let tape = S::new(&ctx, abs_x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&t, [1.0, 5.0]), [1.0, 5.0].into());
        assert_eq!(eval.eval_x(&t, [-2.0, 5.0]), [0.0, 5.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, 5.0]), [0.0, 6.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, -1.0]), [1.0, 6.0].into());

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let tape = S::new(&ctx, sum).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_xy(&t, [0.0, 1.0], [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(
            eval.eval_xy(&t, [1.0, 5.0], [-2.0, 3.0]),
            [1.0, 8.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [1.0, 5.0], [-4.0, 3.0]),
            [1.0, 9.0].into()
        );
    }

    pub fn test_i_sqrt<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let tape = S::new(&ctx, sqrt_x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&t, [0.0, 4.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x(&t, [-2.0, 4.0]), [0.0, 2.0].into());
        let nanan = eval.eval_x(&t, [-2.0, -1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let (v, _) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_square<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let tape = S::new(&ctx, sqrt_x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_x(&t, [0.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x(&t, [2.0, 4.0]), [4.0, 16.0].into());
        assert_eq!(eval.eval_x(&t, [-2.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, -2.0]), [4.0, 36.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, 1.0]), [0.0, 36.0].into());

        let (v, _) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_neg<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let neg_x = ctx.neg(x).unwrap();

        let tape = S::new(&ctx, neg_x).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [-1.0, 0.0].into());
        assert_eq!(eval.eval_x(&t, [0.0, 4.0]), [-4.0, 0.0].into());
        assert_eq!(eval.eval_x(&t, [2.0, 4.0]), [-4.0, -2.0].into());
        assert_eq!(eval.eval_x(&t, [-2.0, 4.0]), [-4.0, 2.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, -2.0]), [2.0, 6.0].into());
        assert_eq!(eval.eval_x(&t, [-6.0, 1.0]), [-1.0, 6.0].into());

        let (v, _) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let tape = S::new(&ctx, mul).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_xy(&t, [0.0, 1.0], [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_xy(&t, [0.0, 1.0], [0.0, 2.0]), [0.0, 2.0].into());
        assert_eq!(
            eval.eval_xy(&t, [-2.0, 1.0], [0.0, 1.0]),
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [-2.0, -1.0], [-5.0, -4.0]),
            [4.0, 10.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [-3.0, -1.0], [-2.0, 6.0]),
            [-18.0, 6.0].into()
        );

        let (v, _) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval(&t, [0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul_imm<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let mul = ctx.mul(x, 2.0).unwrap();
        let tape = S::new(&ctx, mul).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_x(&t, [1.0, 2.0]), [2.0, 4.0].into());

        let mul = ctx.mul(x, -3.0).unwrap();
        let tape = S::new(&ctx, mul).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [-3.0, 0.0].into());
        assert_eq!(eval.eval_x(&t, [1.0, 2.0]), [-6.0, -3.0].into());
    }

    pub fn test_i_sub<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let tape = S::new(&ctx, sub).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(
            eval.eval_xy(&t, [0.0, 1.0], [0.0, 1.0]),
            [-1.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [0.0, 1.0], [0.0, 2.0]),
            [-2.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [-2.0, 1.0], [0.0, 1.0]),
            [-3.0, 1.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [-2.0, -1.0], [-5.0, -4.0]),
            [2.0, 4.0].into()
        );
        assert_eq!(
            eval.eval_xy(&t, [-3.0, -1.0], [-2.0, 6.0]),
            [-9.0, 1.0].into()
        );
    }

    pub fn test_i_sub_imm<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sub = ctx.sub(x, 2.0).unwrap();
        let tape = S::new(&ctx, sub).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [-2.0, -1.0].into());
        assert_eq!(eval.eval_x(&t, [1.0, 2.0]), [-1.0, 0.0].into());

        let sub = ctx.sub(-3.0, x).unwrap();
        let tape = S::new(&ctx, sub).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        assert_eq!(eval.eval_x(&t, [0.0, 1.0]), [-4.0, -3.0].into());
        assert_eq!(eval.eval_x(&t, [1.0, 2.0]), [-5.0, -4.0].into());
    }

    pub fn test_i_recip<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let tape = S::new(&ctx, recip).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();

        let nanan = eval.eval_x(&t, [0.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x(&t, [-1.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_x(&t, [-2.0, 3.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        assert_eq!(eval.eval_x(&t, [-2.0, -1.0]), [-1.0, -0.5].into());
        assert_eq!(eval.eval_x(&t, [1.0, 2.0]), [0.5, 1.0].into());
    }

    pub fn test_i_div<S: ShapeIntervalEval + MathShape>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let div = ctx.div(x, y).unwrap();
        let tape = S::new(&ctx, div).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();

        let nanan = eval.eval_xy(&t, [0.0, 1.0], [-1.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy(&t, [0.0, 1.0], [-2.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_xy(&t, [0.0, 1.0], [0.0, 4.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let out = eval.eval_xy(&t, [-1.0, 0.0], [1.0, 2.0]);
        assert_eq!(out, [-1.0, 0.0].into());

        let out = eval.eval_xy(&t, [-1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, 2.0].into());

        let out = eval.eval_xy(&t, [1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, -1.0].into());

        let out = eval.eval_xy(&t, [-1.0, 4.0], [0.5, 1.0]);
        assert_eq!(out, [-2.0, 8.0].into());

        let (v, _) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let (v, _) = eval
            .eval(&t, [0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_min<S: ShapeIntervalEval + MathShape>()
    where
        <<S as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval
            .eval(&t, [0.0, 1.0], [0.5, 1.5], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&t, [0.0, 1.0], [2.0, 3.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (v, data) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval(&t, [0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());
    }

    pub fn test_i_min_imm<S: ShapeIntervalEval + MathShape>()
    where
        <<S as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) =
            eval.eval(&t, [0.0, 1.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [0.0, 1.0].into());
        assert!(data.is_none());

        let (r, data) =
            eval.eval(&t, [-1.0, 0.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [-1.0, 0.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (r, data) =
            eval.eval(&t, [2.0, 3.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);
    }

    pub fn test_i_max<S: ShapeIntervalEval + MathShape>()
    where
        <<S as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = S::new(&ctx, max).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval
            .eval(&t, [0.0, 1.0], [0.5, 1.5], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [0.5, 1.5].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&t, [0.0, 1.0], [2.0, 3.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (v, data) = eval
            .eval(&t, [std::f32::NAN; 2], [0.0, 1.0], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let (v, data) = eval
            .eval(&t, [0.0, 1.0], [std::f32::NAN; 2], [0.0; 2], &[])
            .unwrap();
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert!(data.is_none());

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let tape = S::new(&ctx, max_xy_z).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 1.0], [4.0, 5.0], &[])
            .unwrap();
        assert_eq!(r, [4.0, 5.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Right]);

        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 1.0], [1.0, 4.0], &[])
            .unwrap();
        assert_eq!(r, [2.0, 4.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Both]);

        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 1.0], [1.0, 1.5], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left, Choice::Left]);
    }

    pub fn test_i_simplify<S: ShapeIntervalEval + MathShape>()
    where
        <<S as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let min = ctx.min(x, 1.0).unwrap();

        let tape = S::new(&ctx, min).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (out, data) =
            eval.eval(&t, [0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 1.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval(&t, [0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [0.0, 0.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);

        let (out, data) =
            eval.eval(&t, [1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let max = ctx.max(x, 1.0).unwrap();
        let tape = S::new(&ctx, max).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (out, data) =
            eval.eval(&t, [0.0, 2.0], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 2.0].into());
        assert!(data.is_none());

        let (out, data) =
            eval.eval(&t, [0.0, 0.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (out, data) =
            eval.eval(&t, [1.5, 2.5], [0.0; 2], [0.0; 2], &[]).unwrap();
        assert_eq!(out, [1.5, 2.5].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_max_imm<S: ShapeIntervalEval + MathShape>()
    where
        <<S as ShapeIntervalEval>::Eval as TracingEvaluator<Interval>>::Trace:
            AsRef<[Choice]>,
    {
        let mut ctx = Context::new();
        let x = ctx.x();
        let max = ctx.max(x, 1.0).unwrap();

        let tape = S::new(&ctx, max).unwrap();
        let t = tape.tape();
        let mut eval = S::Eval::new();
        let (r, data) = eval
            .eval(&t, [0.0, 2.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [1.0, 2.0].into());
        assert!(data.is_none());

        let (r, data) = eval
            .eval(&t, [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [1.0, 1.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Right]);

        let (r, data) = eval
            .eval(&t, [2.0, 3.0], [0.0, 0.0], [0.0, 0.0], &[])
            .unwrap();
        assert_eq!(r, [2.0, 3.0].into());
        assert_eq!(data.unwrap().as_ref(), &[Choice::Left]);
    }

    pub fn test_i_var<S: ShapeIntervalEval + MathShape + ShapeVars>() {
        let mut ctx = Context::new();
        let a = ctx.var("a").unwrap();
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
            2.0.into()
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
            2.0.into()
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
        };
    }
}
