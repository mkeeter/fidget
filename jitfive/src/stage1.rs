use crate::indexed::{define_index, IndexVec};
use crate::stage0::{NodeIndex, Op};

define_index!(
    GroupIndex,
    "Index of a group, globally unique in the compiler pipeline"
);

pub struct Stage1 {
    ops: IndexVec<(Op, GroupIndex), NodeIndex>,
    root: NodeIndex,
    groups: IndexVec<Vec<NodeIndex>, GroupIndex>,
}
