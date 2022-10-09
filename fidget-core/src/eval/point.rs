use crate::{eval::Choice, tape::Tape};

/// Function handle for `f32` evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `PointEval` objects, which actually do evaluation.
pub trait PointFuncT {
    type Evaluator: PointEvalT;
    type Recurse<'a>: PointFuncT;

    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;
    fn get_evaluator(&self) -> Self::Evaluator;
}

/// `f32` evaluator
pub trait PointEvalT {
    fn eval_p(&mut self, x: f32, y: f32, z: f32, c: &mut [Choice]) -> f32;
}

/// Function handle for point evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `PointEval` objects, which actually do evaluation.
pub struct PointFunc<'a, F> {
    tape: &'a Tape,
    func: F,
}

impl<'a, F: PointFuncT<Recurse<'a> = F>> PointFunc<'a, F> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            func: F::from_tape(tape),
        }
    }
    pub fn get_evaluator(&self) -> PointEval<'a, F::Evaluator> {
        PointEval {
            tape: self.tape,
            choices: vec![Choice::Unknown; self.tape.choice_count()],
            eval: self.func.get_evaluator(),
        }
    }
}

pub struct PointEval<'a, E> {
    pub(crate) tape: &'a Tape,
    pub(crate) choices: Vec<Choice>,
    pub(crate) eval: E,
}

impl<'a, E: PointEvalT> PointEval<'a, E> {
    /// Calculates a simplified [`Tape`](crate::tape::Tape) based on the last
    /// evaluation.
    pub fn simplify(&self, reg_limit: u8) -> Tape {
        self.tape.simplify_with_reg_limit(&self.choices, reg_limit)
    }

    /// Resets the internal choice array to `Choice::Unknown`
    fn reset_choices(&mut self) {
        self.choices.fill(Choice::Unknown);
    }

    /// Performs point evaluation
    pub fn eval_p(&mut self, x: f32, y: f32, z: f32) -> f32 {
        self.reset_choices();
        let out = self.eval.eval_p(x, y, z, self.choices.as_mut_slice());
        out
    }
}
