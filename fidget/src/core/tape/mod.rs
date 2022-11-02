//! Infrastructure for building ordered expressions suitable for evaluation
mod builder;
mod op;
mod ssa;
mod tape;

pub(super) use builder::SsaTapeBuilder;
pub(super) use op::TapeOp;
pub(super) use ssa::SsaTape;

pub use tape::{Tape, Workspace};
