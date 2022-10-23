use crate::{
    eval::{Choice, EvalFamily},
    tape::Tape,
};

/// Represents a range, with conservative calculations to guarantee that it
/// always contains the actual value.
///
/// # Warning
/// This implementation does not set rounding modes, so it may not be _perfect_.
#[derive(Copy, Clone, Debug, PartialEq)]
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

/// Trait for interval evaluation, usually wrapped in an
/// [`IntervalEval`](IntervalEval)
pub trait IntervalEvalT: Clone + Send + From<Tape> {
    type Family: EvalFamily;

    fn eval_i<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        choices: &mut [Choice],
    ) -> Interval;
}

#[derive(Clone)]
pub struct IntervalEval<E> {
    tape: Tape,
    choices: Vec<Choice>,
    eval: E,
}

impl<E: IntervalEvalT> From<Tape> for IntervalEval<E> {
    fn from(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            choices: vec![Choice::Unknown; tape.choice_count()],
            eval: E::from(tape),
        }
    }
}

impl<E: IntervalEvalT> IntervalEval<E> {
    pub fn tape(&self) -> Tape {
        self.tape.clone()
    }

    /// Calculates a simplified [`Tape`](crate::tape::Tape) based on the last
    /// evaluation.
    pub fn simplify(&self, reg_limit: u8) -> Tape {
        self.tape.simplify_with_reg_limit(&self.choices, reg_limit)
    }

    /// Resets the internal choice array to `Choice::Unknown`
    fn reset_choices(&mut self) {
        self.choices.fill(Choice::Unknown);
    }

    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }

    /// Performs interval evaluation
    pub fn eval_i<I: Into<Interval>>(&mut self, x: I, y: I, z: I) -> Interval {
        self.reset_choices();
        let out = self.eval.eval_i(x, y, z, self.choices.as_mut_slice());
        out
    }

    /// Performs interval evaluation, using zeros for Y and Z
    ///
    /// This is a convenience function for unit testing
    pub fn eval_i_x<I: Into<Interval>>(&mut self, x: I) -> Interval {
        self.eval_i(x.into(), Interval::new(0.0, 0.0), Interval::new(0.0, 0.0))
    }

    /// Performs interval evaluation, using zeros for Z
    ///
    /// This is a convenience function for unit testing
    pub fn eval_i_xy<I: Into<Interval>>(&mut self, x: I, y: I) -> Interval {
        self.eval_i(x.into(), y.into(), Interval::new(0.0, 0.0))
    }

    /// Evaluates an interval with subdivision, for higher precision
    ///
    /// The given interval is split into `2**subdiv` sub-intervals, then the
    /// resulting bounds are combined.  Running with `subdiv = 0` is equivalent
    /// to calling [`Self::eval_i`].
    ///
    /// This produces a more tightly-bounded accurate result at the cost of
    /// increased computation, but can be a good trade-off if interval
    /// evaluation is cheap!
    pub fn eval_i_subdiv<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        subdiv: usize,
    ) -> Interval {
        self.reset_choices();
        self.eval_subdiv_recurse(x, y, z, subdiv)
    }

    fn eval_subdiv_recurse<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        subdiv: usize,
    ) -> Interval {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        if subdiv == 0 {
            self.eval.eval_i(x, y, z, self.choices.as_mut_slice())
        } else {
            let dx = x.upper() - x.lower();
            let dy = y.upper() - y.lower();
            let dz = z.upper() - z.lower();

            // Helper function to shorten code below
            let mut f = |x: Interval, y: Interval, z: Interval| {
                self.eval_subdiv_recurse(x, y, z, subdiv - 1)
            };

            let (a, b) = if dx >= dy && dx >= dz {
                let x_mid = x.lower() + dx / 2.0;
                (
                    f(Interval::new(x.lower(), x_mid), y, z),
                    f(Interval::new(x_mid, x.upper()), y, z),
                )
            } else if dy >= dz {
                let y_mid = y.lower() + dy / 2.0;
                (
                    f(x, Interval::new(y.lower(), y_mid), z),
                    f(x, Interval::new(y_mid, y.upper()), z),
                )
            } else {
                let z_mid = z.lower() + dz / 2.0;
                (
                    f(x, y, Interval::new(z.lower(), z_mid)),
                    f(x, y, Interval::new(z_mid, z.upper())),
                )
            };
            if a.has_nan() || b.has_nan() {
                std::f32::NAN.into()
            } else {
                Interval::new(
                    a.lower().min(b.lower()),
                    a.upper().max(b.upper()),
                )
            }
        }
    }
}

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
    use crate::context::Context;

    pub fn test_interval<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = ctx.get_tape(x, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [2.0, 3.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_i_xy([1.0, 5.0], [2.0, 3.0]), [1.0, 5.0].into());

        let tape = ctx.get_tape(y, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [2.0, 3.0]), [2.0, 3.0].into());
        assert_eq!(eval.eval_i_xy([1.0, 5.0], [4.0, 5.0]), [4.0, 5.0].into());
    }

    pub fn test_i_abs<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let tape = ctx.get_tape(abs_x, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_i_x([1.0, 5.0]), [1.0, 5.0].into());
        assert_eq!(eval.eval_i_x([-2.0, 5.0]), [0.0, 5.0].into());
        assert_eq!(eval.eval_i_x([-6.0, 5.0]), [0.0, 6.0].into());
        assert_eq!(eval.eval_i_x([-6.0, -1.0]), [1.0, 6.0].into());

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let tape = ctx.get_tape(sum, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_i_xy([1.0, 5.0], [-2.0, 3.0]), [1.0, 8.0].into());
        assert_eq!(eval.eval_i_xy([1.0, 5.0], [-4.0, 3.0]), [1.0, 9.0].into());
    }

    pub fn test_i_sqrt<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let tape = ctx.get_tape(sqrt_x, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_i_x([0.0, 4.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_i_x([-2.0, 4.0]), [0.0, 2.0].into());
        let nanan = eval.eval_i_x([-2.0, -1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_square<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let tape = ctx.get_tape(sqrt_x, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_i_x([0.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_i_x([2.0, 4.0]), [4.0, 16.0].into());
        assert_eq!(eval.eval_i_x([-2.0, 4.0]), [0.0, 16.0].into());
        assert_eq!(eval.eval_i_x([-6.0, -2.0]), [4.0, 36.0].into());
        assert_eq!(eval.eval_i_x([-6.0, 1.0]), [0.0, 36.0].into());

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let tape = ctx.get_tape(mul, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 1.0].into());
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [0.0, 2.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_i_xy([-2.0, 1.0], [0.0, 1.0]), [-2.0, 1.0].into());
        assert_eq!(
            eval.eval_i_xy([-2.0, -1.0], [-5.0, -4.0]),
            [4.0, 10.0].into()
        );
        assert_eq!(
            eval.eval_i_xy([-3.0, -1.0], [-2.0, 6.0]),
            [-18.0, 6.0].into()
        );

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let v = eval.eval_i([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_mul_imm<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let two = ctx.constant(2.0);
        let mul = ctx.mul(x, two).unwrap();
        let tape = ctx.get_tape(mul, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [0.0, 2.0].into());
        assert_eq!(eval.eval_i_x([1.0, 2.0]), [2.0, 4.0].into());

        let neg_three = ctx.constant(-3.0);
        let mul = ctx.mul(x, neg_three).unwrap();
        let tape = ctx.get_tape(mul, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [-3.0, 0.0].into());
        assert_eq!(eval.eval_i_x([1.0, 2.0]), [-6.0, -3.0].into());
    }

    pub fn test_i_sub<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let tape = ctx.get_tape(sub, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [0.0, 1.0]), [-1.0, 1.0].into());
        assert_eq!(eval.eval_i_xy([0.0, 1.0], [0.0, 2.0]), [-2.0, 1.0].into());
        assert_eq!(eval.eval_i_xy([-2.0, 1.0], [0.0, 1.0]), [-3.0, 1.0].into());
        assert_eq!(
            eval.eval_i_xy([-2.0, -1.0], [-5.0, -4.0]),
            [2.0, 4.0].into()
        );
        assert_eq!(
            eval.eval_i_xy([-3.0, -1.0], [-2.0, 6.0]),
            [-9.0, 1.0].into()
        );
    }

    pub fn test_i_sub_imm<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let two = ctx.constant(2.0);
        let sub = ctx.sub(x, two).unwrap();
        let tape = ctx.get_tape(sub, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [-2.0, -1.0].into());
        assert_eq!(eval.eval_i_x([1.0, 2.0]), [-1.0, 0.0].into());

        let neg_three = ctx.constant(-3.0);
        let sub = ctx.sub(neg_three, x).unwrap();
        let tape = ctx.get_tape(sub, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(eval.eval_i_x([0.0, 1.0]), [-4.0, -3.0].into());
        assert_eq!(eval.eval_i_x([1.0, 2.0]), [-5.0, -4.0].into());
    }

    pub fn test_i_recip<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let tape = ctx.get_tape(recip, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);

        let nanan = eval.eval_i_x([0.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_i_x([-1.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_i_x([-2.0, 3.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        assert_eq!(eval.eval_i_x([-2.0, -1.0]), [-1.0, -0.5].into());
        assert_eq!(eval.eval_i_x([1.0, 2.0]), [0.5, 1.0].into());
    }

    pub fn test_i_div<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let div = ctx.div(x, y).unwrap();
        let tape = ctx.get_tape(div, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);

        let nanan = eval.eval_i_xy([0.0, 1.0], [-1.0, 1.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_i_xy([0.0, 1.0], [-2.0, 0.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let nanan = eval.eval_i_xy([0.0, 1.0], [0.0, 4.0]);
        assert!(nanan.lower().is_nan());
        assert!(nanan.upper().is_nan());

        let out = eval.eval_i_xy([-1.0, 0.0], [1.0, 2.0]);
        assert_eq!(out, [-1.0, 0.0].into());

        let out = eval.eval_i_xy([-1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, 2.0].into());

        let out = eval.eval_i_xy([1.0, 4.0], [-1.0, -0.5]);
        assert_eq!(out, [-8.0, -1.0].into());

        let out = eval.eval_i_xy([-1.0, 4.0], [0.5, 1.0]);
        assert_eq!(out, [-2.0, 8.0].into());

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());

        let v = eval.eval_i([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
    }

    pub fn test_i_min<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(
            eval.eval_i([0.0, 1.0], [0.5, 1.5], [0.0; 2]),
            [0.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(
            eval.eval_i([0.0, 1.0], [2.0, 3.0], [0.0; 2]),
            [0.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 1.0], [0.0; 2]),
            [0.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Right]);

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);

        let v = eval.eval_i([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);
    }

    pub fn test_i_min_imm<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();

        let tape = ctx.get_tape(min, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(
            eval.eval_i([0.0, 1.0], [0.0; 2], [0.0; 2]),
            [0.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(
            eval.eval_i([-1.0, 0.0], [0.0; 2], [0.0; 2]),
            [-1.0, 0.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0; 2], [0.0; 2]),
            [1.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Right]);
    }

    pub fn test_i_max<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = ctx.get_tape(max, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(
            eval.eval_i([0.0, 1.0], [0.5, 1.5], [0.0; 2],),
            [0.5, 1.5].into()
        );
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(
            eval.eval_i([0.0, 1.0], [2.0, 3.0], [0.0; 2]),
            [2.0, 3.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Right]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 1.0], [0.0; 2],),
            [2.0, 3.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left]);

        let v = eval.eval_i([std::f32::NAN; 2], [0.0, 1.0], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);

        let v = eval.eval_i([0.0, 1.0], [std::f32::NAN; 2], [0.0; 2]);
        assert!(v.lower().is_nan());
        assert!(v.upper().is_nan());
        assert_eq!(eval.choices(), &[Choice::Both]);

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let tape = ctx.get_tape(max_xy_z, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 1.0], [4.0, 5.0]),
            [4.0, 5.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left, Choice::Right]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 1.0], [1.0, 4.0]),
            [2.0, 4.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left, Choice::Both]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 1.0], [1.0, 1.5]),
            [2.0, 3.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left, Choice::Left]);
    }

    pub fn test_i_max_imm<I: IntervalEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let one = ctx.constant(1.0);
        let max = ctx.max(x, one).unwrap();

        let tape = ctx.get_tape(max, I::Family::REG_LIMIT);
        let mut eval = IntervalEval::<I>::from(tape);
        assert_eq!(
            eval.eval_i([0.0, 2.0], [0.0, 0.0], [0.0, 0.0]),
            [1.0, 2.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Both]);

        assert_eq!(
            eval.eval_i([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
            [1.0, 1.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Right]);

        assert_eq!(
            eval.eval_i([2.0, 3.0], [0.0, 0.0], [0.0, 0.0]),
            [2.0, 3.0].into()
        );
        assert_eq!(eval.choices(), &[Choice::Left]);
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
        };
    }
}
