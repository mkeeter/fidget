//! Tools for working with virtual assembly code
mod alloc;
mod asm_op;
mod lru;

pub(super) use alloc::RegisterAllocator;

pub use asm_op::AsmOp;

// TODO move this to its own file, rename to VmTape?
#[derive(Clone)]
pub struct AsmTape {
    tape: Vec<AsmOp>,
    slot_count: usize,
}

impl AsmTape {
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
    pub fn new(tape: Vec<AsmOp>, slot_count: usize) -> Self {
        Self { tape, slot_count }
    }
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, AsmOp> {
        self.tape.iter()
    }
}
