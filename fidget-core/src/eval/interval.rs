use crate::{eval::Choice, tape::Tape};

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
}

impl From<[f32; 2]> for Interval {
    fn from(i: [f32; 2]) -> Interval {
        Interval::new(i[0], i[1])
    }
}

impl Interval {
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
        if self.lower > 0.0 {
            Interval::new(1.0 / self.upper, 1.0 / self.lower)
        } else if self.upper < 0.0 {
            Interval::new(1.0 / self.lower, 1.0 / self.upper)
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

pub struct IntervalEval<E> {
    pub(crate) tape: Tape,
    pub(crate) choices: Vec<Choice>,
    pub(crate) eval: E,
}

impl<E: IntervalEvalT> IntervalEval<E> {
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

/// Trait for a function handle stored in a [`IntervalFunc`](IntervalFunc)
pub trait IntervalFuncT: Sync {
    type Evaluator: IntervalEvalT;

    fn from_tape(tape: Tape) -> Self;

    /// Return the evaluator type, which may borrow from this `struct`
    ///
    /// This should be an O(1) operation; heavy lifting should have been
    /// previously done when constructing the `IntervalFuncT` itself.
    fn get_evaluator(&self) -> Self::Evaluator;
}

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `IntervalEval` objects, which actually do evaluation.
pub struct IntervalFunc<F> {
    tape: Tape,
    func: F,
}

impl<F: IntervalFuncT> IntervalFunc<F> {
    pub fn tape(&self) -> Tape {
        self.tape.clone()
    }

    pub fn from_tape(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            func: F::from_tape(tape),
        }
    }

    pub fn get_evaluator(&self) -> IntervalEval<F::Evaluator> {
        IntervalEval {
            tape: self.tape.clone(),
            choices: vec![Choice::Unknown; self.tape.choice_count()],
            eval: self.func.get_evaluator(),
        }
    }
}

/// Trait for interval evaluation, usually wrapped in an
/// [`IntervalEval`](IntervalEval)
///
/// The evaluator will likely have a lifetime bounded to its parent
/// [`IntervalFuncT`](IntervalFuncT), and can generate
/// a new [`Tape`](crate::tape::Tape) on demand after evaluation.
pub trait IntervalEvalT {
    fn eval_i<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        choices: &mut [Choice],
    ) -> Interval;
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
