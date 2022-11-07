//! Tools for working with tapes in single static assignment (SSA) form
mod builder;
mod op;
mod tape;

pub use builder::Builder;
pub use op::Op;
pub use tape::Tape;
