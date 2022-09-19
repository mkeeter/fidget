use crate::{eval::Interval, tape::Tape};

pub trait EvalSeed<'a> {
    type IntervalFunc: IntervalFunc<'a>;
    type FloatSliceFunc: FloatSliceFunc<'a>;
    fn from_tape_i(t: &'a Tape) -> Self::IntervalFunc;
    fn from_tape_s(t: &'a Tape) -> Self::FloatSliceFunc;
}

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `IntervalEval` objects, which actually do evaluation.
pub trait IntervalFunc<'a>: Sync {
    type Evaluator: IntervalEval<'a>;

    fn get_evaluator(&self) -> Self::Evaluator;
}

/// Interval evaluator
///
/// The evaluator will likely have a lifetime bounded to its parent
/// [`IntervalFunc`](crate::eval::IntervalFunc), and can generate
/// a new [`Tape`](crate::tape::Tape) on demand after evaluation.
pub trait IntervalEval<'a> {
    fn simplify(&self) -> Tape;

    /// Evaluates the given interval and records choices into the internal
    /// array, _without_ resetting the choice array beforehand.
    ///
    /// This function is a building block for `eval_subdiv`, but likely
    /// shouldn't be called on its own.
    #[doc(hidden)]
    fn eval_i_inner<I: Into<Interval>>(&mut self, x: I, y: I, z: I)
        -> Interval;

    /// Resets the internal choice array to `Choice::Unknown`
    #[doc(hidden)]
    fn reset_choices(&mut self);

    /// Post-evaluation wrangling of choice array
    #[doc(hidden)]
    fn load_choices(&mut self);

    /// Performs interval evaluation
    fn eval_i<I: Into<Interval>>(&mut self, x: I, y: I, z: I) -> Interval {
        self.reset_choices();
        let out = self.eval_i_inner(x, y, z);
        self.load_choices();
        out
    }

    /// Performs interval evaluation
    fn eval_i_x<I: Into<Interval>>(&mut self, x: I) -> Interval {
        self.eval_i(x.into(), Interval::new(0.0, 0.0), Interval::new(0.0, 0.0))
    }

    /// Performs interval evaluation
    fn eval_i_xy<I: Into<Interval>>(&mut self, x: I, y: I) -> Interval {
        self.eval_i(x.into(), y.into(), Interval::new(0.0, 0.0))
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
    ) -> Interval {
        self.reset_choices();
        let out = self.eval_subdiv_recurse(x, y, z, subdiv.saturating_sub(1));
        self.load_choices();
        out
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
            Interval::new(a.lower().min(b.lower()), a.upper().max(b.upper()))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Function handle for evaluation of many points simultaneously.
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub trait FloatSliceFunc<'a> {
    type Evaluator: FloatSliceEval<'a>;
    type Recurse<'b>: FloatSliceFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator;
}

/// Simultaneous evaluation of many points
pub trait FloatSliceEval<'a> {
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
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

impl<'a, F: FloatSliceEval<'a>> FloatEval<'a> for F {
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [0.0];
        self.eval_s(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
