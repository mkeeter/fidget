//! Instruction tapes in the form of assembly for a simple virtual machine
mod active;
mod alloc;
mod choice;
mod eval;
mod lru;
mod op;
mod tape;

pub(crate) mod build;

pub use choice::{ChoiceIndex, ChoiceMask, Choices};
pub use eval::Eval;
pub use op::Op;
pub use tape::{ChoiceTape, Tape, TapeData, TapeSpecialization};
