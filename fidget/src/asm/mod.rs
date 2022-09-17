//! Tools for working with virtual and machine assembly code
mod alloc;
mod asm_op;
mod lru;

pub mod dynasm;

pub(crate) use alloc::RegisterAllocator;

pub use asm_op::AsmOp;
