//! Traits and data structures for evaluation
//!
//! The easiest way to build an evaluator of a particular kind is the
//! [`Eval`](Eval) extension trait on [`Family`](Family):
//!
//! ```rust
//! use fidget::eval::Eval;
//! use fidget::vm;
//! use fidget::context::Context;
//!
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let tape = ctx.get_tape(x).unwrap();
//!
//! // `vm::Eval` implements `Family`, so we can use it to build any kind of
//! // evaluator.  In this case, we'll build a single-point evaluator:
//! let mut eval = vm::Eval::new_point_evaluator(tape);
//! assert_eq!(eval.eval_p(0.25, 0.0, 0.0, &[]).unwrap(), 0.25);
//! ```

pub mod float_slice;
pub mod grad;
pub mod interval;
pub mod point;
pub mod tape;

mod choice;
mod vars;

// Re-export a few things
pub use choice::Choice;
pub use float_slice::FloatSliceEval;
pub use grad::GradEval;
pub use interval::Interval;
pub use point::PointEval;
pub use tape::Tape;
pub use vars::Vars;

use float_slice::FloatSliceEvalT;
use grad::GradEvalT;
use point::PointEvalT;

/// Represents a "family" of evaluators (JIT, interpreter, etc)
pub trait Family: Clone {
    /// Register limit for this evaluator family.
    const REG_LIMIT: u8;

    type IntervalEval: TracingEvaluator<crate::eval::interval::Interval, Self>
        + EvaluatorStorage<Self>
        + Clone
        + Send;
    type FloatSliceEval: FloatSliceEvalT<Self>;
    type PointEval: PointEvalT<Self>;
    type GradEval: GradEvalT<Self>;

    /// Recommended tile sizes for 3D rendering
    fn tile_sizes_3d() -> &'static [usize];

    /// Recommended tile sizes for 2D rendering
    fn tile_sizes_2d() -> &'static [usize];
}

/// Helper trait used to add evaluator constructions to anything implementing
/// [`Family`](Family).
pub trait Eval<F: Family> {
    fn new_point_evaluator(tape: Tape<F>) -> point::PointEval<F>;
    fn new_interval_evaluator(
        tape: Tape<F>,
    ) -> TracingEval<Interval, F::IntervalEval, F>;
    fn new_interval_evaluator_with_storage(
        tape: Tape<F>,
        storage: <<F as Family>::IntervalEval as EvaluatorStorage<F>>::Storage,
    ) -> TracingEval<Interval, F::IntervalEval, F>;
    fn new_float_slice_evaluator(
        tape: Tape<F>,
    ) -> float_slice::FloatSliceEval<F>;

    fn new_float_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: float_slice::FloatSliceEvalStorage<F>,
    ) -> float_slice::FloatSliceEval<F>;

    fn new_grad_evaluator(tape: Tape<F>) -> grad::GradEval<F>;

    fn new_grad_evaluator_with_storage(
        tape: Tape<F>,
        storage: grad::GradEvalStorage<F>,
    ) -> grad::GradEval<F>;
}

impl<F: Family> Eval<F> for F {
    /// Builds a point evaluator from the given `Tape`
    fn new_point_evaluator(tape: Tape<F>) -> point::PointEval<F> {
        point::PointEval::new(tape)
    }

    /// Builds an interval evaluator from the given `Tape`
    fn new_interval_evaluator(
        tape: Tape<F>,
    ) -> TracingEval<Interval, F::IntervalEval, F> {
        TracingEval::new(&tape)
    }

    /// Builds an interval evaluator from the given `Tape`, reusing storage
    fn new_interval_evaluator_with_storage(
        tape: Tape<F>,
        storage: <<F as Family>::IntervalEval as EvaluatorStorage<F>>::Storage,
    ) -> TracingEval<Interval, F::IntervalEval, F> {
        TracingEval::new_with_storage(&tape, storage)
    }

    /// Builds a float evaluator from the given `Tape`
    fn new_float_slice_evaluator(
        tape: Tape<F>,
    ) -> float_slice::FloatSliceEval<F> {
        float_slice::FloatSliceEval::new(tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_float_slice_evaluator_with_storage(
        tape: Tape<F>,
        storage: float_slice::FloatSliceEvalStorage<F>,
    ) -> float_slice::FloatSliceEval<F> {
        float_slice::FloatSliceEval::new_with_storage(tape, storage)
    }

    /// Builds a grad slice evaluator from the given `Tape`
    fn new_grad_evaluator(tape: Tape<F>) -> grad::GradEval<F> {
        grad::GradEval::new(tape)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    fn new_grad_evaluator_with_storage(
        tape: Tape<F>,
        storage: grad::GradEvalStorage<F>,
    ) -> grad::GradEval<F> {
        grad::GradEval::new_with_storage(tape, storage)
    }
}

////////////////////////////////////////////////////////////////////////////////
// NEW STUFF

use crate::Error;

pub trait EvaluatorStorage<F> {
    type Storage: Default;

    /// Constructs the evaluator, giving it a chance to reuse storage
    ///
    /// The incoming `Storage` is consumed, though it may not necessarily be
    /// used to construct the new tape (e.g. if it's a memory-mapped region and
    /// is too small).
    fn new_with_storage(tape: &Tape<F>, storage: Self::Storage) -> Self;

    /// Extract the internal storage for reuse, if possible
    fn take(self) -> Option<Self::Storage>;
}

/// A tracing evaluator performs evaluation of a single `T`, capturing a trace
/// of execution for further simplification.
///
/// This trait is unlikely to be used directly; instead, use a
/// [`TracingEval`](TracingEval), which offers a user-friendly API.
pub trait TracingEvaluator<T, F> {
    /// Workspace type used during evaluation
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

/// Generic tracing evaluator
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

    _p: std::marker::PhantomData<*const T>,
}

unsafe impl<T, E: Send, F> Send for TracingEval<T, E, F> {}

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
pub struct TracingEvalData<D, F> {
    choices: Vec<Choice>,

    /// Inner data
    data: D,

    /// Stores the most recent tape and whether simplification is allowed
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

    // TODO move these to a separate result type
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
