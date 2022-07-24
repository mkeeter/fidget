use std::collections::BTreeSet;

use crate::{
    indexed::{define_index, IndexMap, IndexVec},
    op::GenericOp,
};

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

pub type Op = GenericOp<VarIndex, f64, NodeIndex, ChoiceIndex>;

/// Stage 0 of the shape compiler pipeline.
///
/// Operations are tightly packed and assigned a globally unique index.
pub struct Stage0 {
    /// Unordered list of operations.
    ops: IndexVec<Op, NodeIndex>,

    /// Root operation in the tree
    root: NodeIndex,

    /// Number of nodes in the tree which make LHS/RHS choices
    num_choices: usize,

    /// Bi-directional map of variable names to indexes
    vars: IndexMap<String, VarIndex>,
}

impl Stage0 {
    pub fn self_check(&self) {
        assert!(usize::from(self.root) < self.ops.len());
        let mut used = BTreeSet::new();
        // TODO: implement IntoIter on IndexVec?
        for o in self.ops.iter() {
            for a in o.iter_children() {
                used.insert(a);
                assert!(
                    usize::from(a) < self.ops.len(),
                    "Invalid child node index"
                );
            }
            match o {
                Op::Var(v) => {
                    assert!(
                        usize::from(*v) < self.vars.len(),
                        "Invalid var index"
                    );
                }
                Op::Max(_, _, c) | Op::Min(_, _, c) => {
                    assert!(
                        usize::from(*c) < self.num_choices,
                        "Invalid choice index"
                    );
                }
                _ => (),
            }
        }
        assert!(
            used.len() == self.ops.len() - 1,
            "All nodes must be used at least once"
        );

        used.insert(self.root);
        assert!(
            used.len() == self.ops.len(),
            "The root cannot be used in the graph"
        );
    }
}
