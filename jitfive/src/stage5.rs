use crate::{
    indexed::{IndexMap, IndexVec},
    stage0::{NodeIndex, Op, VarIndex},
    stage1::{GroupIndex, TaggedOp},
    stage4::{Group, Stage4},
};

/// Stores a graph of math expressions, a graph of node groups, and a tree of
/// node groups (based on lowest common ancestors in the graph).
///
/// Unlike `Stage2`, `groups[i].children` and `groups[i].nodes` are
/// topologically sorted, so they can be evaluated (recursively) in order.
#[derive(Debug)]
pub struct Stage5 {
    /// Math operations, stored in arbitrary order and associated with a group
    pub ops: IndexVec<TaggedOp, NodeIndex>,

    /// Root of the tree
    pub root: NodeIndex,

    /// Groups of nodes and group graph links, stored in arbitrary order
    pub groups: IndexVec<Group, GroupIndex>,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,

    /// Stores the last time when a node is used.
    ///
    /// The `usize` is based on an in-order traversal, iterating over groups
    /// (recursively) then nodes within a group in sorted order.
    pub last_use: IndexVec<usize, NodeIndex>,
}

fn recurse(
    g: GroupIndex,
    t: &Stage4,
    i: &mut usize,
    last_used: &mut IndexVec<usize, NodeIndex>,
) {
    let group = &t.groups[g];
    for &cg in &group.children {
        recurse(cg, t, i, last_used);
    }
    for &n in &group.nodes {
        let op = t.ops[n];
        match op.op {
            Op::Var(..) | Op::Const(..) => (),
            Op::Binary(_op, a, b) => {
                last_used[a] = *i;
                last_used[b] = *i;
            }
            Op::BinaryChoice(_op, a, b, ..) => {
                last_used[a] = *i;
                last_used[b] = *i;
            }
            Op::Unary(_op, a) => {
                last_used[a] = *i;
            }
        }
        *i += 1;
    }
}

impl From<&Stage4> for Stage5 {
    fn from(t: &Stage4) -> Self {
        let mut last_use = IndexVec::new();
        last_use.resize(t.ops.len(), 0);
        let mut i = 0;
        recurse(t.ops[t.root].group, t, &mut i, &mut last_use);

        Self {
            ops: t.ops.clone(),
            root: t.root,
            groups: t.groups.clone(),
            num_choices: t.num_choices,
            vars: t.vars.clone(),
            last_use,
        }
    }
}
