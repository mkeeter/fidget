//! Tape used for evaluation
use crate::compiler::{RegOp, RegisterAllocator, SsaTape};
use serde::{Deserialize, Serialize};

/// Low-level tape for use with the Fidget virtual machine (or to be lowered
/// further into machine instructions).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct RegTape {
    tape: Vec<RegOp>,

    /// Total allocated slots
    pub(super) slot_count: u32,

    /// Number of variables
    pub(super) var_count: u32,
}

impl RegTape {
    /// Lowers the tape to assembly with a particular register limit
    ///
    /// Note that if you _also_ want to simplify the tape, it's more efficient
    /// to use [`VmData::simplify`](crate::vm::VmData::simplify), which
    /// simultaneously simplifies **and** performs register allocation in a
    /// single pass.
    pub fn new<const N: usize>(ssa: &SsaTape) -> Self {
        let mut alloc = RegisterAllocator::<N>::new(ssa.len());
        for &op in ssa.iter() {
            alloc.op(op)
        }
        alloc.finalize()
    }

    /// Builds a new empty tape, with one allocated slot
    pub(crate) fn empty() -> Self {
        Self {
            tape: vec![],
            slot_count: 1,
            var_count: 0,
        }
    }

    /// Resets this tape, retaining its allocations
    pub fn reset(&mut self) {
        self.tape.clear();
        self.slot_count = 1;
    }
    /// Returns the number of unique register and memory locations that are used
    /// by this tape.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count as usize
    }
    /// Returns the number of variables (inputs) used in this tape
    #[inline]
    pub fn var_count(&self) -> usize {
        self.var_count as usize
    }
    /// Returns the number of elements in the tape
    #[inline]
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    /// Returns `true` if the tape contains no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }
    /// Returns a front-to-back iterator
    ///
    /// This is the opposite of evaluation order; it will visit the root of the
    /// tree first, and end at the leaves.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, RegOp> {
        self.into_iter()
    }
    #[inline]
    pub(crate) fn push(&mut self, op: RegOp) {
        self.tape.push(op)
    }
}

impl<'a> IntoIterator for &'a RegTape {
    type Item = &'a RegOp;
    type IntoIter = std::slice::Iter<'a, RegOp>;
    fn into_iter(self) -> Self::IntoIter {
        self.tape.iter()
    }
}
