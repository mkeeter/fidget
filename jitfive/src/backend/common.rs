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
