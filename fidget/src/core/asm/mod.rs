//! Tools for working with virtual assembly code
mod alloc;
mod asm_op;
mod lru;

pub(super) use alloc::RegisterAllocator;

pub use asm_op::AsmOp;

// TODO move this to its own file, rename to VmTape?
#[derive(Clone, Default)]
pub struct AsmTape {
    tape: Vec<AsmOp>,

    /// Total allocated slots
    slot_count: u32,

    /// Number of registers, before we fall back to Load/Store operations
    reg_limit: u8,
}

impl AsmTape {
    pub fn new(reg_limit: u8) -> Self {
        Self {
            tape: Vec::with_capacity(512),
            slot_count: 1,
            reg_limit,
        }
    }
    pub fn reset(&mut self, reg_limit: u8) {
        self.tape.clear();
        self.slot_count = 1;
        self.reg_limit = reg_limit;
    }
    pub fn reg_limit(&self) -> u8 {
        self.reg_limit
    }
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
    pub fn iter(&self) -> std::slice::Iter<'_, AsmOp> {
        self.tape.iter()
    }
    #[inline]
    pub fn push(&mut self, op: AsmOp) {
        self.tape.push(op)
    }
}
