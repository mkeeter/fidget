//! Tools for working with virtual and machine assembly code
mod alloc;
mod asm_eval;
mod asm_op;
mod choice;
mod lru;

pub mod dynasm;

pub use alloc::RegisterAllocator;
pub use asm_eval::AsmEval;
pub use asm_op::AsmOp;
pub use choice::Choice;
