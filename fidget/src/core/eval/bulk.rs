//! Evaluates many points in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! A bulk evaluator expects to be given **many single points**, i.e. the X, Y,
//! Z inputs are always `&[f32]`.  The output may be of a different type, e.g.
//! partial derivatives with respect to X/Y/Z
//! ([`GradEval`](crate::eval::GradSliceEval)).
//!
//! Bulk evaluators are typically named `XSliceEval`, where `X` is the output
//! type.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{
    eval::{EvaluatorStorage, Family},
    vm::{Choices, Tape, TapeData},
    Error,
};

/// Trait for bulk evaluation returning the given type `T`
///
/// It's uncommon to use this trait outside the library itself; it's an
/// abstraction to reduce code duplication, and is public because it's used as a
/// constraint on other public APIs.
pub trait BulkEvaluator<T, F: Family> {
    /// Data type used during evaluation
    ///
    /// For example, an interpreter would put its intermediate slot storage into
    /// its `Data` type, so it could be reused.
    type Data: BulkEvaluatorData<F> + Default + Send;

    /// Evaluates many points, writing the result into `out` and using `data` as
    /// scratch memory.
    ///
    /// # Panics
    /// This function may assume that the `x`, `y`, `z`, and `out` slices are of
    /// equal length and panic otherwise; higher-level calls should maintain
    /// that invariant.
    ///
    /// This function may also assume that `vars` is correctly sized for the
    /// number of variables in the tape.
    fn eval_with(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
        out: &mut [T],
        choices: &mut Choices,
        data: &mut Self::Data,
    );
}

/// Trait for data associated with a particular bulk evaluator.
pub trait BulkEvaluatorData<F: Family> {
    /// Prepares the given data structure to be used for evaluation of the
    /// specified tape with the specified number of items.
    ///
    /// This is vague; as a specific example, it may include resizing internal
    /// data arrays based on the tape's slot count.
    fn prepare(&mut self, tape: &TapeData<F>, size: usize);
}

/// Some bulk evaluators have no need for scratch data!
impl<F: Family> BulkEvaluatorData<F> for () {
    fn prepare(&mut self, _tape: &TapeData<F>, _size: usize) {
        // Nothing to do here
    }
}

/// Generic bulk evaluator container `struct`
///
/// This includes an inner type implementing
/// [`BulkEvaluator`](BulkEvaluator) and a stored [`Tape`](Tape).
///
/// This type is parameterized with three types:
/// - `T` is the output type returned by bulk evaluation
/// - `E` is the bulk evaluator itself
/// - `F` is the tape family
///
/// The internal `tape` is planned with
/// [`F::REG_LIMIT`](crate::eval::Family::REG_LIMIT) registers.
#[derive(Clone)]
pub struct BulkEval<T, E, F: Family> {
    eval: E,
    tape: Tape<F>,

    _p: std::marker::PhantomData<fn(T) -> T>,
}

impl<T, E, F: Family> BulkEval<T, E, F>
where
    E: BulkEvaluator<T, F> + EvaluatorStorage<F>,
    T: Clone + From<f32>,
{
    /// Builds a new evaluator for the given tape, allocating new storage
    pub fn new(tape: Tape<F>) -> Self {
        Self::new_with_storage(tape, E::Storage::default())
    }

    /// Returns a copy of the inner tape
    pub fn tape(&self) -> Tape<F> {
        self.tape.clone()
    }

    /// Builds a new evaluator for the given tape, reusing the given storage
    pub fn new_with_storage(tape: Tape<F>, storage: E::Storage) -> Self {
        let eval = E::new_with_storage(&tape, storage);
        Self {
            eval,
            tape,
            _p: std::marker::PhantomData,
        }
    }

    /// Consumes the evaluator, returning the inner storage type for reuse
    pub fn take(self) -> Option<E::Storage> {
        self.eval.take()
    }

    /// Evaluate using the given `data` as scratch memory
    ///
    /// Returns a slice of results borrowed from `data.out`.
    pub fn eval_with<'a>(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
        data: &'a mut BulkEvalData<E::Data, T, F>,
    ) -> Result<&'a [T], Error> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(Error::MismatchedSlices);
        }
        let expected_var_count = self.tape.data().var_count();
        if vars.len() != expected_var_count {
            return Err(Error::BadVarSlice(vars.len(), expected_var_count));
        }
        data.prepare(self.tape.data(), x.len());
        self.eval.eval_with(
            x,
            y,
            z,
            vars,
            &mut data.out,
            &mut data.choices,
            &mut data.data,
        );
        Ok(&data.out)
    }

    /// Evaluates the given slices, returning a fresh `Vec<T>`
    ///
    /// This function performs allocation; in a hot loop, consider using
    /// [`eval_with`](Self::eval_with) instead.
    pub fn eval(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        vars: &[f32],
    ) -> Result<Vec<T>, Error> {
        let mut data = Default::default();
        self.eval_with(x, y, z, vars, &mut data)?;
        Ok(data.out)
    }
}

/// Generic data associated with a bulk evaluator
///
/// This data is used during evaluator.
///
/// - `D` is the scratch (mutable) data type used by the evaluator
/// - `T` is the evaluation value type (e.g. `f32`)
/// - `F` is the tape family
pub struct BulkEvalData<D, T, F> {
    out: Vec<T>,

    /// Choice data array
    choices: Choices,

    /// Inner data
    data: D,

    _p: std::marker::PhantomData<fn() -> F>,
}

impl<D: Default, T, F> Default for BulkEvalData<D, T, F> {
    fn default() -> Self {
        Self {
            out: vec![],
            choices: Choices::default(),
            data: D::default(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<D: BulkEvaluatorData<F>, T, F> BulkEvalData<D, T, F>
where
    T: Clone + From<f32>,
    F: Family,
{
    fn prepare(&mut self, tape: &TapeData<F>, size: usize) {
        self.out.resize(size, std::f32::NAN.into());
        self.out.fill(std::f32::NAN.into());
        self.choices.resize_and_zero(tape.choice_array_size());
        self.data.prepare(tape, size);
    }
}
