//! Tape used for evaluation
use crate::vm::Op;

/// Low-level tape for use with the Fidget virtual machine (or to be lowered
/// further into machine instructions).
#[derive(Clone, Default)]
pub struct Tape {
    tape: Vec<Op>,

    /// Total allocated slots
    pub(super) slot_count: u32,

    /// Number of registers, before we fall back to Load/Store operations
    reg_limit: u8,
}

impl Tape {
    pub fn new(reg_limit: u8) -> Self {
        Self {
            tape: vec![],
            slot_count: 1,
            reg_limit,
        }
    }
    /// Resets this tape, retaining its allocations
    pub fn reset(&mut self, reg_limit: u8) {
        self.tape.clear();
        self.slot_count = 1;
        self.reg_limit = reg_limit;
    }
    /// Returns the register limit with which this tape was planned
    pub fn reg_limit(&self) -> u8 {
        self.reg_limit
    }
    /// Returns the number of unique register and memory locations that are used
    /// by this tape.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count as usize
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Op> {
        self.tape.iter()
    }
    #[inline]
    pub(crate) fn push(&mut self, op: Op) {
        self.tape.push(op)
    }
}
