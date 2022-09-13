//! Infrastructure for building ordered expressions suitable for evaluation
mod builder;
mod op;
mod ssa;
mod tape;

pub use builder::SsaTapeBuilder;
pub use op::TapeOp;
pub use ssa::SsaTape;
pub use tape::Tape;
