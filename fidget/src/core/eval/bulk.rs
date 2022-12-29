//! Evaluates many values in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{
    eval::{EvaluatorStorage, Family, Tape},
    Error,
};

/// Trait for bulk evaluation of a given type
pub trait BulkEvaluator<T, F> {
    type Data: BulkEvaluatorData<F> + Default;

    /// Evaluates a slice of `T`, writing the result into `out`
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
        x: &[T],
        y: &[T],
        z: &[T],
        vars: &[f32],
        out: &mut [T],
        data: &mut Self::Data,
    );
}

/// Trait for data associated with a particular bulk evaluator.
pub trait BulkEvaluatorData<F> {
    /// Prepares the given data structure to be used for evaluation of the
    /// specified tape with the specified number of items.
    ///
    /// This is vague; as a specific example, it may include resizing internal
    /// data arrays based on the tape's slot count.
    fn prepare(&mut self, tape: &Tape<F>, size: usize);
}

impl<F> BulkEvaluatorData<F> for () {
    fn prepare(&mut self, _tape: &Tape<F>, _size: usize) {
        // Nothing to do here
    }
}

/// Generic bulk evaluator `struct`
///
/// This includes an inner type implementing
/// [`BulkEvaluator`](BulkEvaluator) and a stored [`Tape`](Tape).
///
/// The internal `tape` is planned with
/// [`E::REG_LIMIT`](crate::eval::Family::REG_LIMIT) registers.
#[derive(Clone)]
pub struct BulkEval<T, E, F> {
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
    ///
    /// Returns a slice of results borrowed from `data.out`.
    pub fn eval_with<'a>(
        &self,
        x: &[T],
        y: &[T],
        z: &[T],
        vars: &[f32],
        data: &'a mut BulkEvalData<E::Data, T, F>,
    ) -> Result<&'a [T], Error> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(Error::MismatchedSlices);
        } else if vars.len() != self.tape.var_count() {
            return Err(Error::BadVarSlice(vars.len(), self.tape.var_count()));
        }
        data.prepare(&self.tape, x.len());
        self.eval
            .eval_with(x, y, z, vars, &mut data.out, &mut data.data);
        Ok(&data.out)
    }

    /// Evaluates the given slices, returning a fresh `Vec<T>`
    ///
    /// This function performs allocation; in a hot loop, consider using
    /// [`eval_with`](Self::eval_with) instead.
    pub fn eval(
        &self,
        x: &[T],
        y: &[T],
        z: &[T],
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

    /// Inner data
    data: D,

    _p: std::marker::PhantomData<*const F>,
}

impl<D: Default, T, F> Default for BulkEvalData<D, T, F> {
    fn default() -> Self {
        Self {
            out: vec![],
            data: D::default(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<D: BulkEvaluatorData<F>, T, F> BulkEvalData<D, T, F>
where
    T: Clone + From<f32>,
{
    fn prepare(&mut self, tape: &Tape<F>, size: usize) {
        self.out.resize(size, std::f32::NAN.into());
        self.out.fill(std::f32::NAN.into());
        self.data.prepare(tape, size);
    }
}
