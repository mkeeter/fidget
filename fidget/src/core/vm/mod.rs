//! Instruction tapes in the form of assembly for a simple virtual machine
mod alloc;
mod eval;
mod op;

pub(crate) mod lru; // TODO
pub(crate) mod tape; // TODO

pub(super) use alloc::RegisterAllocator;

pub use eval::Eval;
pub use op::Op;
pub use tape::Tape;
