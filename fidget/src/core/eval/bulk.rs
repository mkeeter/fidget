//! Evaluates many points in a single call
//!
//! Doing bulk evaluations helps limit to overhead of instruction dispatch, and
//! can take advantage of SIMD.
//!
//! It is unlikely that you'll want to use these traits or types directly;
//! they're implementation details to minimize code duplication.

use crate::{eval::Tape, Error};

/// Trait for bulk evaluation returning the given type `T`
///
/// Bulk evaluators should usually be constructed on a per-thread basis.
///
/// They contain (at minimum) output array storage, which is borrowed in the
/// return from [`eval`](BulkEvaluator::eval).  They may also contain
/// intermediate storage (e.g. an array of VM registers).
pub trait BulkEvaluator: Default {
    /// Data type used during evaluation
    type Data: From<f32> + Copy + Clone;

    /// Instruction tape used during evaluation
    ///
    /// This may be a literal instruction tape (in the case of VM evaluation),
    /// or a metaphorical instruction tape (e.g. a JIT function).
    type Tape: Tape<Storage = Self::TapeStorage> + Send + Sync;

    /// Associated type for tape storage
    ///
    /// This is a workaround for plumbing purposes
    type TapeStorage;

    /// Evaluates many points using the given instruction tape
    ///
    /// `vars` should be a slice-of-slices (or a slice-of-`Vec`s) representing
    /// input arguments for each of the tape's variables; use [`Tape::vars`] to
    /// map from [`Var`](crate::var::Var) to position in the list.
    ///
    /// The returned slice is borrowed from the evaluator.
    ///
    /// Returns an error if any of the `var` slices are of different lengths, or
    /// if all variables aren't present.
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<Self::Data>, Error>;

    /// Build a new empty evaluator
    fn new() -> Self {
        Self::default()
    }
}

/// Container for bulk output results
///
/// This container represents an array-of-arrays.  It is indexed first by
/// output index, then by index within the evaluation array.
pub struct BulkOutput<'a, T> {
    data: &'a Vec<Vec<T>>,
    len: usize,
}

impl<'a, T> BulkOutput<'a, T> {
    pub(crate) fn new(data: &'a Vec<Vec<T>>, len: usize) -> Self {
        Self { data, len }
    }

    /// Returns the number of output variables
    ///
    /// Note that this is **not** the length of each individual output slice;
    /// that can be found with `out[0].len()` (assuming there is at least one
    /// output variable).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks whether the output contains zero variables
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T> std::ops::Index<usize> for BulkOutput<'a, T> {
    type Output = [T];
    fn index(&self, i: usize) -> &'a Self::Output {
        &self.data[i][0..self.len]
    }
}

impl<'a, T> BulkOutput<'a, T> {
    /// Helper function to borrow using the original reference lifetime
    pub(crate) fn borrow(&self, i: usize) -> &'a [T] {
        &self.data[i][0..self.len]
    }
}
