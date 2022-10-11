//! Tools for working with virtual assembly code
mod alloc;
mod asm_op;
mod lru;

pub(crate) use alloc::RegisterAllocator;

pub use asm_op::AsmOp;
