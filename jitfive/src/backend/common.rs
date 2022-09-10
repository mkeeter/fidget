use crate::{op::GenericOp, util::indexed::define_index};

define_index!(
    VarIndex,
    "Index of a variable, globally unique in the compiler pipeline"
);
define_index!(
    NodeIndex,
    "Index of a node, globally unique in the compiler pipeline"
);

pub type Op = GenericOp<VarIndex, f64, NodeIndex>;

/// Represents a single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
///
/// Note that this is a bitfield such that
/// ```text
/// Choice::Both = Choice::Left | Choice::Right
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    Unknown = 0,
    Left = 1,
    Right = 2,
    Both = 3,
}

pub trait Simplify {
    fn simplify(&self, choices: &[Choice]) -> Self;
}
