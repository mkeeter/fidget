//! Interval evaluation
use crate::eval::{
    tracing::{TracingEval, TracingEvalData, TracingEvaluator},
    Choice, EvaluatorStorage, Family,
};

/// Represents a range, with conservative calculations to guarantee that it
/// always contains the actual value.
///
/// # Warning
/// This implementation does not set rounding modes, so it may not be _perfect_.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Interval {
    lower: f32,
    upper: f32,
}

impl Interval {
    #[inline]
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(upper >= lower || (lower.is_nan() && upper.is_nan()));
        Self { lower, upper }
    }
    #[inline]
    pub fn lower(&self) -> f32 {
        self.lower
    }
    #[inline]
    pub fn upper(&self) -> f32 {
        self.upper
    }
    pub fn has_nan(&self) -> bool {
        self.lower.is_nan() || self.upper.is_nan()
    }
    pub fn abs(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval::new(0.0, self.upper.max(-self.lower))
            } else {
                Interval::new(-self.upper, -self.lower)
            }
        } else {
            self
        }
    }
    pub fn square(self) -> Self {
        if self.upper < 0.0 {
            Interval::new(self.upper.powi(2), self.lower.powi(2))
        } else if self.lower > 0.0 {
            Interval::new(self.lower.powi(2), self.upper.powi(2))
        } else if self.has_nan() {
            std::f32::NAN.into()
        } else {
            Interval::new(0.0, self.lower.abs().max(self.upper.abs()).powi(2))
        }
    }
    pub fn sqrt(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval::new(0.0, self.upper.sqrt())
            } else {
                std::f32::NAN.into()
            }
        } else {
            Interval::new(self.lower.sqrt(), self.upper.sqrt())
        }
    }
    pub fn recip(self) -> Self {
        if self.lower > 0.0 || self.upper < 0.0 {
            Interval::new(1.0 / self.upper, 1.0 / self.lower)
        } else {
            std::f32::NAN.into()
        }
    }
    pub fn min_choice(self, rhs: Self) -> (Self, Choice) {
        if self.has_nan() || rhs.has_nan() {
            return (std::f32::NAN.into(), Choice::Both);
        }
        let choice = if self.upper < rhs.lower {
            Choice::Left
        } else if rhs.upper < self.lower {
            Choice::Right
        } else {
            Choice::Both
        };
        (
            Interval::new(self.lower.min(rhs.lower), self.upper.min(rhs.upper)),
            choice,
        )
    }
    pub fn max_choice(self, rhs: Self) -> (Self, Choice) {
        if self.has_nan() || rhs.has_nan() {
            return (std::f32::NAN.into(), Choice::Both);
        }
        let choice = if self.lower > rhs.upper {
            Choice::Left
        } else if rhs.lower > self.upper {
            Choice::Right
        } else {
            Choice::Both
        };
        (
            Interval::new(self.lower.max(rhs.lower), self.upper.max(rhs.upper)),
            choice,
        )
    }
}

impl From<[f32; 2]> for Interval {
    fn from(i: [f32; 2]) -> Interval {
        Interval::new(i[0], i[1])
    }
}

impl From<f32> for Interval {
    fn from(f: f32) -> Self {
        Interval::new(f, f)
    }
}

impl std::ops::Add<Interval> for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Interval::new(self.lower + rhs.lower, self.upper + rhs.upper)
    }
}

impl std::ops::Mul<Interval> for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self.has_nan() || rhs.has_nan() {
            return std::f32::NAN.into();
        }
        let mut out = [0.0; 4];
        let mut k = 0;
        for i in [self.lower, self.upper] {
            for j in [rhs.lower, rhs.upper] {
                out[k] = i * j;
                k += 1;
            }
        }
        let mut lower = out[0];
        let mut upper = out[0];
        for &v in &out[1..] {
            lower = lower.min(v);
            upper = upper.max(v);
        }
        Interval::new(lower, upper)
    }
}

impl std::ops::Div<Interval> for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if self.has_nan() {
            return std::f32::NAN.into();
        }
        if rhs.lower > 0.0 || rhs.upper < 0.0 {
            let mut out = [0.0; 4];
            let mut k = 0;
            for i in [self.lower, self.upper] {
                for j in [rhs.lower, rhs.upper] {
                    out[k] = i / j;
                    k += 1;
                }
            }
            let mut lower = out[0];
            let mut upper = out[0];
            for &v in &out[1..] {
                lower = lower.min(v);
                upper = upper.max(v);
            }
            Interval::new(lower, upper)
        } else {
            std::f32::NAN.into()
        }
    }
}

impl std::ops::Sub<Interval> for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Interval::new(self.lower - rhs.upper, self.upper - rhs.lower)
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Interval::new(-self.upper, -self.lower)
    }
}

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
        eval::{Eval, Vars},
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
