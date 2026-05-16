//! Tape used for evaluation
use crate::compiler::{RegOp, RegisterAllocator, SsaTape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Low-level tape for use with the Fidget virtual machine (or to be lowered
/// further into machine instructions).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct RegTape {
    tape: Vec<RegOp>,

    /// Total allocated slots
    ///
    /// This is a continuous space of registers (`0..N`) and memory (`N..`),
    /// where `N` is the parameter in [`RegTape::new`].
    pub(super) slot_count: u32,
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

    /// Repacks registers by frequency (so that register 0 is the most frequent)
    pub fn repack(&mut self) {
        let map = self.repack_map();
        for op in &mut self.tape {
            op.visit_regs_mut(|reg| *reg = map[reg]);
        }
    }

    /// Returns a map for register repacking
    ///
    /// The map repacks registers in the tape by frequency, so that register 0
    /// is the most frequent.
    pub fn repack_map(&self) -> HashMap<u8, u8> {
        let mut reg_counts: HashMap<u8, usize> = HashMap::new();
        for op in &self.tape {
            op.visit_regs(|reg| *reg_counts.entry(reg).or_default() += 1);
        }
        let mut sorted = reg_counts
            .into_iter()
            .map(|(reg, count)| (std::cmp::Reverse(count), reg))
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        sorted
            .into_iter()
            .enumerate()
            .map(|(i, (_count, reg))| (reg, u8::try_from(i).unwrap()))
            .collect()
    }

    /// Builds a new empty tape
    pub(crate) fn empty() -> Self {
        Self {
            tape: vec![],
            slot_count: 0,
        }
    }

    /// Resets this tape, retaining its allocations
    pub fn reset(&mut self) {
        self.tape.clear();
        self.slot_count = 0;
    }

    /// Returns the number of unique register and memory locations that are used
    /// by this tape.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count as usize
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
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &RegOp> {
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
