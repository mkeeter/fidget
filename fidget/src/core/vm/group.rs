use crate::{
    compiler::{RegTape, SsaOp},
    vm::Choice,
};
use std::sync::Arc;

struct GroupData {
    /// Operations in this group, in reverse-evaluation order
    ops: Vec<SsaOp>,

    /// Offset of this group's first choice in a global choice array
    choice_offset: usize,

    /// Subsequent groups which are always enabled if this group is active
    enable_always: Vec<usize>,

    /// Choice data, in reverse-evaluation order
    choices: Vec<ChoiceData>,
}

struct ChoiceData {
    /// Index of the operation to be patched in the `Vec<SsaOp>`
    op_index: usize,

    /// Group to enable if the choice includes `Choice::Left`
    enable_left: usize,

    /// Group to enable if the choice includes `Choice::Right`
    enable_right: usize,
}

struct GroupTape {
    root: Arc<Vec<GroupData>>,
    choice: Vec<Choice>,
    active_groups: Vec<usize>,
    tape: RegTape,
}
