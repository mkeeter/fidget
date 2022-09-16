//! Infrastructure for building ordered expressions suitable for evaluation
mod builder;
mod op;
mod ssa;
mod tape;

pub(crate) use builder::SsaTapeBuilder;
pub(crate) use op::TapeOp;
pub(crate) use ssa::SsaTape;

pub use tape::Tape;
