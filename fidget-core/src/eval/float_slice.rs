use crate::tape::Tape;

/// Function handle for evaluation of many points simultaneously.
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub trait FloatSliceFuncT {
    type Evaluator: FloatSliceEvalT;

    /// Type system workaround for recursion
    ///
    /// This should be identical to the type on which we're implementing
    /// `FloatSliceFuncT`, but with an arbitrary lifetime attached
    type Recurse<'a>: FloatSliceFuncT;

    /// Storage used by the type
    type Storage;

    fn from_tape(tape: &Tape) -> Self::Recurse<'_>;

    /// Constructs the `FloatSliceFuncT`, giving it a chance to reuse storage
    ///
    /// If the `storage` argument is used, then it's consumed; otherwise, it's
    /// returned as part of the tuple.
    fn from_tape_give(
        tape: &Tape,
        storage: Self::Storage,
    ) -> (Self::Recurse<'_>, Option<Self::Storage>);

    /// Returns an evaluator, which may borrow from this handle
    ///
    /// This should be an O(1) operation; heavy lifting should have been
    /// previously done when constructing the `FloatSliceFuncT` itself.
    fn get_evaluator(&self) -> Self::Evaluator;

    /// Extract the internal storage for reuse
    fn take(self) -> Self::Storage;

    /// Erases the lifetime from a nested `Storage` object
    ///
    /// This is a lifetime workaround that should be equivalent to
    /// [`std::convert::identity`].
    fn lift(
        s: <Self::Recurse<'_> as FloatSliceFuncT>::Storage,
    ) -> Self::Storage;
}

/// Simultaneous evaluation of many points
pub trait FloatSliceEvalT {
    fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]);
}

/// Function handle for interval evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `FloatSliceEval` objects, which actually do evaluation.
pub struct FloatSliceFunc<'a, F: FloatSliceFuncT> {
    tape: &'a Tape,
    func: F::Recurse<'a>,
}

impl<'a, F: FloatSliceFuncT> FloatSliceFunc<'a, F> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            func: F::from_tape(tape),
        }
    }
    pub fn new_give(
        tape: &'a Tape,
        s: F::Storage,
    ) -> (Self, Option<F::Storage>) {
        let (func, out) = F::from_tape_give(tape, s);
        (Self { tape, func }, out)
    }

    pub fn get_evaluator(
        &self,
    ) -> FloatSliceEval<'a, <F::Recurse<'a> as FloatSliceFuncT>::Evaluator>
    {
        FloatSliceEval {
            tape: self.tape,
            eval: self.func.get_evaluator(),
        }
    }
    pub fn take(self) -> <F::Recurse<'a> as FloatSliceFuncT>::Storage {
        self.func.take()
    }
}

pub struct FloatSliceEval<'a, E> {
    #[allow(dead_code)]
    pub(crate) tape: &'a Tape,
    pub(crate) eval: E,
}

impl<'a, E: FloatSliceEvalT> FloatSliceEval<'a, E> {
    pub fn eval_s(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [f32]) {
        self.eval.eval_s(x, y, z, out)
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut out = [std::f32::NAN];
        self.eval_s(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
