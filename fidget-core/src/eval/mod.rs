//! Evaluation, both generically and with a small local interpreter

pub mod asm;
mod choice;

pub mod float_slice;
pub mod interval;
pub mod point;

// Re-export a few things
pub(crate) use choice::Choice;
