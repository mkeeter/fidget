//! Infrastructure for representing math expressions as graphs
mod context;
mod indexed;
mod op;

pub use context::{Context, Node, VarNode};
pub use op::{BinaryOpcode, Op, UnaryOpcode};
