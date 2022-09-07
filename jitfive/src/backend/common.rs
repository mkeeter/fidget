use crate::{op::GenericOp, util::indexed::define_index};

define_index!(
    VarIndex,
    "Index of a variable, globally unique in the compiler pipeline"
);
define_index!(
    NodeIndex,
    "Index of a node, globally unique in the compiler pipeline"
);
define_index!(
    ChoiceIndex,
    "Index of a choice, globally unique in the compiler pipeline"
);
define_index!(
    GroupIndex,
    "Index of a group, globally unique in the compiler pipeline"
);

pub type Op = GenericOp<VarIndex, f64, NodeIndex, ChoiceIndex>;

/// Represents a single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    Left,
    Right,
    Both,
}

pub trait Simplify {
    fn simplify(&self, choices: &[Choice]) -> Self;
}
