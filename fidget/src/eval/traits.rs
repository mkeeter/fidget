use crate::{eval::Interval, tape::Tape};

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `IntervalEval` objects, which actually do evaluation.
pub trait IntervalFunc<'a>: Sync {
    type Evaluator: IntervalEval<'a>;
    type Recurse<'b>: IntervalFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// Token produced by interval evaluation, which allows you to simplify a `Tape`
pub struct EvalToken<'a, 'b, I: IntervalEval<'a> + ?Sized>(
    &'b mut I,
    std::marker::PhantomData<&'a ()>,
);
impl<'a, 'b, I: IntervalEval<'a>> EvalToken<'a, 'b, I> {
    /// Returns a tape that only includes active clauses
    pub fn simplify(self) -> Tape {
        self.0.simplify()
    }
}

pub(crate) mod private {
    use super::*;
    #[doc(hidden)]
    pub trait Simplify {
        fn simplify(&self) -> Tape;
    }
}

/// Interval evaluator
///
/// The evaluator will likely have a lifetime bounded to its parent
/// [`IntervalFunc`](crate::eval::IntervalFunc), and can generate
/// a new [`Tape`](crate::tape::Tape) on demand after evaluation.
pub trait IntervalEval<'a>: private::Simplify {
    /// Evaluates the given interval and records choices into the internal
    /// array, _without_ resetting the choice array beforehand.
    ///
    /// This function is a building block for `eval_subdiv`, but likely
    /// shouldn't be called on its own.
    fn eval_i_inner<I: Into<Interval>>(&mut self, x: I, y: I, z: I)
        -> Interval;

    /// Resets the internal choice array to `Choice::Unknown`
    fn reset_choices(&mut self);

    /// Performs interval evaluation and tape simplification
    fn eval_i<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
    ) -> (Interval, EvalToken<'a, '_, Self>) {
        self.reset_choices();
        let out = self.eval_i_inner(x, y, z);
        (out, EvalToken(self, std::marker::PhantomData))
    }

    /// Evaluates an interval with subdivision
    ///
    /// The given interval is split into `2**subdiv` sub-intervals, then the
    /// resulting bounds are combined.  Running with `subdiv = 0` or `subdiv =
    /// 1` is equivalent to calling [`Self::eval_i`].
    ///
    /// This produces a more tightly-bounded accurate result at the cost of
    /// increased computation, but can be a good trade-off if interval
    /// evaluation is cheap!
    fn eval_i_subdiv<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        subdiv: usize,
    ) -> (Interval, EvalToken<'a, '_, Self>) {
        self.reset_choices();
        let out = self.eval_subdiv_recurse(x, y, z, subdiv.saturating_sub(1));
        (out, EvalToken(self, std::marker::PhantomData))
    }

    #[doc(hidden)]
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
            self.eval_i_inner(x, y, z)
        } else {
            let dx = x.upper - x.lower;
            let dy = y.upper - y.lower;
            let dz = z.upper - z.lower;

            // Helper function to shorten code below
            let mut f = |x: Interval, y: Interval, z: Interval| {
                self.eval_subdiv_recurse(x, y, z, subdiv - 1)
            };

            let (a, b) = if dx >= dy && dx >= dz {
                let x_mid = x.lower + dx / 2.0;
                (
                    f(Interval::new(x.lower, x_mid), y, z),
                    f(Interval::new(x_mid, x.upper), y, z),
                )
            } else if dy >= dz {
                let y_mid = y.lower + dy / 2.0;
                (
                    f(x, Interval::new(y.lower, y_mid), z),
                    f(x, Interval::new(y_mid, y.upper), z),
                )
            } else {
                let z_mid = z.lower + dz / 2.0;
                (
                    f(x, y, Interval::new(z.lower, z_mid)),
                    f(x, y, Interval::new(z_mid, z.upper)),
                )
            };
            Interval::new(a.lower.min(b.lower), a.upper.max(b.upper))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for `f32 x 4` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `VecEval` objects, which actually do evaluation.
pub trait VecFunc<'a> {
    type Evaluator: VecEval<'a>;
    type Recurse<'b>: VecFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// `f32 x 4` evaluator
pub trait VecEval<'a> {
    fn eval_v(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4];
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for `f32` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatEval` objects, which actually do evaluation.
pub trait FloatFunc<'a> {
    type Evaluator: FloatEval<'a>;
    type Recurse<'b>: FloatFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
}

/// `f32` evaluator
pub trait FloatEval<'a> {
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32;
}

impl<'a, F: FloatEval<'a>> VecEval<'a> for F {
    fn eval_v(&mut self, x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = FloatEval::eval_f(self, x[i], y[i], z[i])
        }
        out
    }
}
