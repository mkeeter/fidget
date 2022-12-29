//! Capturing a trace of function evaluation for further optimization
//!
//! Tracing evaluators are run on a single data type and capture a trace of
//! execution, recording decision points in a [`Vec<Choice>`](Choice).  Decision
//! points are places where the code could take a single branch out of multiple
//! options; for example, a `min` or `max` node.
//!
//! The resulting trace can be used to simplify the instruction tape.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{
    eval::{EvaluatorStorage, Family, Tape},
    Error,
};

/// Represents a single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
///
/// Note that this is a bitfield such that
/// ```text
/// Choice::Both = Choice::Left | Choice::Right
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    Unknown = 0,
    Left = 1,
    Right = 2,
    Both = 3,
}

impl std::ops::BitOrAssign<Choice> for Choice {
    fn bitor_assign(&mut self, other: Self) {
        *self = match (*self as u8) | (other as u8) {
            0 => Self::Unknown,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Both,
            _ => unreachable!(),
        }
    }
}

/// A tracing evaluator performs evaluation of a single `T`, capturing a trace
/// of execution for further simplification.
///
/// This trait is unlikely to be used directly; instead, use a
/// [`TracingEval`](TracingEval), which offers a user-friendly API.
pub trait TracingEvaluator<T, F> {
    /// Scratch (mutable) data used during evaluation
    type Data: TracingEvaluatorData<F> + Default;

    /// Evaluates the given value, using `choices` and `data` as scratch memory.
    ///
    /// # Panics
    /// If `vars` or `choices` is of the incorrect length, this function is
    /// allowed (encouraged, even) to panic.
    fn eval_with(
        &self,
        x: T,
        y: T,
        z: T,
        vars: &[f32],
        choices: &mut [Choice],
        data: &mut Self::Data,
    ) -> (T, bool);
}

/// Trait for data associated with a particular tracing evaluator.
pub trait TracingEvaluatorData<F> {
    /// Prepares the given data structure to be used for evaluation of the
    /// specified tape.
    ///
    /// This is vague; as a specific example, it may include resizing internal
    /// data arrays based on the tape's slot count.
    fn prepare(&mut self, tape: &Tape<F>);
}

impl<F> TracingEvaluatorData<F> for () {
    fn prepare(&mut self, _tape: &Tape<F>) {
        // Nothing to do here
    }
}

/// Generic tracing evaluator `struct`
///
/// This includes an inner type implementing
/// [`TracingEvaluator`](TracingEvaluator) and a stored [`Tape`](Tape).
///
/// The internal `tape` is planned with
/// [`E::REG_LIMIT`](crate::eval::Family::REG_LIMIT) registers.
#[derive(Clone)]
pub struct TracingEval<T, E, F> {
    eval: E,
    tape: Tape<F>,

    _p: std::marker::PhantomData<fn(T) -> T>,
}

impl<T, E, F: Family> TracingEval<T, E, F>
where
    E: TracingEvaluator<T, F> + EvaluatorStorage<F>,
{
    /// Builds a new evaluator for the given tape, allocating new storage
    pub fn new(tape: &Tape<F>) -> Self {
        Self::new_with_storage(tape, E::Storage::default())
    }

    pub fn tape(&self) -> Tape<F> {
        self.tape.clone()
    }

    /// Builds a new evaluator for the given tape, reusing the given storage
    pub fn new_with_storage(tape: &Tape<F>, storage: E::Storage) -> Self {
        let eval = E::new_with_storage(tape, storage);
        Self {
            eval,
            tape: tape.clone(),
            _p: std::marker::PhantomData,
        }
    }

    /// Consumes the evaluator, returning the inner storage type for reuse
    pub fn take(self) -> Option<E::Storage> {
        self.eval.take()
    }

    /// Evaluate using (and modifying) the given workspace
    pub fn eval_with<J: Into<T>>(
        &self,
        x: J,
        y: J,
        z: J,
        vars: &[f32],
        data: &mut TracingEvalData<E::Data, F>,
    ) -> Result<T, Error> {
        if vars.len() != self.tape.var_count() {
            return Err(Error::BadVarSlice(vars.len(), self.tape.var_count()));
        }
        data.prepare(&self.tape);
        let (value, simplify) = self.eval.eval_with(
            x.into(),
            y.into(),
            z.into(),
            vars,
            &mut data.choices,
            &mut data.data,
        );
        data.prev = Some((self.tape.clone(), simplify));
        Ok(value)
    }

    pub fn eval<J: Into<T>>(
        &self,
        x: J,
        y: J,
        z: J,
        vars: &[f32],
    ) -> Result<(T, TracingEvalData<E::Data, F>), Error> {
        let mut data = Default::default();
        let out = self.eval_with(x, y, z, vars, &mut data)?;
        Ok((out, data))
    }
}

impl<T, E, F: Family> TracingEval<T, E, F>
where
    E: TracingEvaluator<T, F> + EvaluatorStorage<F>,
    T: From<f32>,
{
    /// Performs interval evaluation, using zeros for Y and Z and no `vars`
    ///
    /// This is a convenience function for unit testing
    pub fn eval_x<J: Into<T>>(&self, x: J) -> T {
        let mut data = Default::default();
        self.eval_with(x.into(), T::from(0.0), T::from(0.0), &[], &mut data)
            .unwrap()
    }

    /// Performs interval evaluation, using zeros for Y and Z and no `vars`
    ///
    /// This is a convenience function for unit testing
    pub fn eval_xy<J: Into<T>>(&self, x: J, y: J) -> T {
        let mut data = Default::default();
        self.eval_with(x.into(), y.into(), T::from(0.0), &[], &mut data)
            .unwrap()
    }
}

/// Generic data associated with a tracing evaluator
///
/// This data is used during evaluator.
pub struct TracingEvalData<D, F> {
    choices: Vec<Choice>,

    /// Inner data
    data: D,

    /// Stores the most recent tape and whether simplification is allowed
    /// TODO: move this to a separate type?
    prev: Option<(Tape<F>, bool)>,
}

impl<D: Default, F> Default for TracingEvalData<D, F> {
    fn default() -> Self {
        Self {
            choices: vec![],
            data: D::default(),
            prev: None,
        }
    }
}

impl<D: TracingEvaluatorData<F>, F: Family> TracingEvalData<D, F> {
    fn prepare(&mut self, tape: &Tape<F>) {
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.choices.fill(Choice::Unknown);
        self.data.prepare(tape);
    }

    // TODO move these to a separate result type?
    pub fn tape_len(&self) -> Option<usize> {
        self.prev.as_ref().map(|t| t.0.len())
    }

    pub fn tape(&self) -> Option<Tape<F>> {
        self.prev.as_ref().map(|t| t.0.clone())
    }

    pub fn should_simplify(&self) -> Option<bool> {
        self.prev.as_ref().map(|t| t.1)
    }

    /// Simplifies the tape based on the most recent evaluation
    ///
    /// Returns an error if no evaluation has been performed.
    pub fn simplify(&self) -> Result<Tape<F>, Error> {
        self.simplify_with(&mut Default::default(), Default::default())
    }

    /// Returns a read-only view into the [`Choice`](Choice) slice.
    ///
    /// This is a convenience function for unit testing.
    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }

    /// Simplifies the tape based on the most recent evaluation, reusing
    /// allocations to reduce memory churn.
    ///
    /// Returns an error if no evaluation has been performed.
    pub fn simplify_with(
        &self,
        workspace: &mut crate::eval::tape::Workspace,
        prev: crate::eval::tape::Data,
    ) -> Result<Tape<F>, Error> {
        if let Some((tape, simplify)) = &self.prev {
            if *simplify {
                tape.simplify_with(&self.choices, workspace, prev)
            } else {
                Ok(tape.clone())
            }
        } else {
            Err(Error::NoTrace)
        }
    }
}
