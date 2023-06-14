//! Instruction tapes in the form of assembly for a simple virtual machine
mod alloc;
mod choice;
mod eval;
mod lru;
mod op;
mod tape;

pub(crate) mod build;

pub use choice::{ChoiceIndex, Choices};
pub use eval::Eval;
pub use op::Op;
pub use tape::{Tape, TapeData, TapeSpecialization};
