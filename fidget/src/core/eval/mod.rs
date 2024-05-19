//! Traits and data structures for function evaluation

#[cfg(any(test, feature = "eval-tests"))]
pub mod test;

/// A tape represents something that can be evaluated by an evaluator
///
/// The only property enforced on the trait is that we must have some way to
/// recycle its internal storage.  This matters most for JIT evaluators, whose
/// tapes are regions of executable memory-mapped RAM (which is expensive to map
/// and unmap).
pub trait Tape {
    /// Associated type for this tape's data storage
    type Storage: Default;

    /// Retrieves the internal storage from this tape
    fn recycle(self) -> Self::Storage;
}

/// Represents the trace captured by a tracing evaluation
///
/// The only property enforced on the trait is that we must have a way of
/// reusing trace allocations.  Because [`Trace`] implies `Clone` where it's
/// used in [`Function`], this is trivial, but we can't provide a default
/// implementation because it would fall afoul of `impl` specialization.
pub trait Trace {
    /// Copies the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);
}

impl<T: Copy + Clone + Default> Trace for Vec<T> {
    fn copy_from(&mut self, other: &Self) {
        self.resize(other.len(), T::default());
        self.copy_from_slice(other);
    }
}
