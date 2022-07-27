use std::collections::BTreeSet;

use crate::indexed::{IndexMap, IndexVec};
use crate::stage0::{NodeIndex, Op, VarIndex};
use crate::stage1::{GroupIndex, Source, Stage1};

/// A group represents a set of nodes which are enabled by the same set
/// of choices at `min` or `max` nodes.
///
/// This `Group` (unlike [`crate::stage1::Group`]) includes graph connections to
/// upstream and downstream groups.
#[derive(Default)]
pub struct Group {
    /// Choices which enable this group of nodes.
    ///
    /// If any choice in this array is valid, then the nodes of the group are
    /// enabled.  Choices are expressed in the positive form ("if choice _i_
    /// is *Left*, then the group is enabled").
    ///
    /// This array is expected to be sorted and unique, since it is used
    /// as a key when collecting nodes into groups.
    pub choices: Vec<Source>,

    /// Nodes in this group, in arbitrary order
    ///
    /// Indexes refer to nodes in the parent stage's `ops` array
    pub nodes: Vec<NodeIndex>,

    /// Downstream groups are farther from the root of the tree
    pub downstream: Vec<GroupIndex>,

    /// Upstream groups are closer to the root of the tree
    pub upstream: Vec<GroupIndex>,
}

/// Stores a graph of math expressions and a graph of node groups
pub struct Stage2 {
    /// Math operations, stored in arbitrary order and associated with a group
    pub ops: IndexVec<(Op, GroupIndex), NodeIndex>,

    /// Root of the tree
    pub root: NodeIndex,

    /// Groups of nodes and group graph links, stored in arbitrary order
    pub groups: IndexVec<Group, GroupIndex>,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,
}

impl From<&Stage1> for Stage2 {
    fn from(t: &Stage1) -> Self {
        let mut downstream: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
            IndexVec::new();
        downstream.resize_with(t.groups.len(), BTreeSet::new);

        let mut upstream: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
            IndexVec::new();
        upstream.resize_with(t.groups.len(), BTreeSet::new);

        // Find group inputs and outputs by noticing cases where a child node
        // is stored in a different group than its caller.
        for (group_index, group) in t.groups.enumerate() {
            for n in group.nodes.iter() {
                for c in t.ops[*n].0.iter_children() {
                    let child_group = t.ops[c].1;
                    if child_group != group_index {
                        downstream[group_index].insert(child_group);
                        upstream[child_group].insert(group_index);
                    }
                }
            }
        }

        let groups = t
            .groups
            .iter()
            .zip(downstream.into_iter().zip(upstream.into_iter()))
            .map(|(g, (downstream, upstream))| Group {
                choices: g.choices.clone(),
                nodes: g.nodes.clone(),
                upstream: upstream.into_iter().collect(),
                downstream: downstream.into_iter().collect(),
            })
            .collect();
        Self {
            ops: t.ops.clone(),
            root: t.root,
            groups,
            num_choices: t.num_choices,
            vars: t.vars.clone(),
        }
    }
}
