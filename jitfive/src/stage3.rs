use std::collections::{BTreeMap, BTreeSet};

use crate::indexed::{IndexMap, IndexVec};
use crate::stage0::{NodeIndex, Op, VarIndex};
use crate::stage1::{GroupIndex, Source};
use crate::stage2::Stage2;

/// A group represents a set of nodes which are enabled by the same set
/// of choices at `min` or `max` nodes.
///
/// This `Group` (unlike [`crate::stage1::Group`]) includes graph connections to
/// upstream and downstream groups.
///
/// In addition, unlike [`crate::stage2::Group`], it includes **tree**
/// connections to parent and child groups.  The tree parent is based on the
/// least common ancestor of all upstream groups.
#[derive(Default, Debug)]
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

    /// Parent of this group in the group tree, based on lowest common ancestor
    /// of the upstream nodes.
    pub parent: Option<GroupIndex>,

    /// Children of this group in the group tree; the opposite of `parent`
    pub children: Vec<GroupIndex>,
}

/// Stores a graph of math expressions, a graph of node groups, and a tree of
/// node groups (based on lowest common ancestors in the graph).
#[derive(Debug)]
pub struct Stage3 {
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

impl From<&Stage2> for Stage3 {
    fn from(t: &Stage2) -> Self {
        let mut ranks = IndexVec::new();
        ranks.resize(t.groups.len(), None);

        let mut ancestors = IndexVec::new();
        ancestors.resize(t.groups.len(), None);
        for g in 0..t.groups.len() {
            populate_ancestors(t, GroupIndex::from(g), &mut ancestors);
        }

        let root_group_index = t.ops[t.root].1;
        populate_ranks(t, root_group_index, 0, &mut ranks);

        let parents: IndexVec<Option<GroupIndex>, GroupIndex> = t
            .groups
            .iter()
            .map(|g| {
                // Special case for the root of the tree
                if g.upstream.is_empty() {
                    return None;
                }
                // The lowest common ancestor is reachable from every upstream
                // group of the target, which we check by counting how many
                // upstream groups can reach each potential common ancestor.
                let mut common_count: BTreeMap<_, usize> = BTreeMap::new();
                for upstream in &g.upstream {
                    for a in ancestors[*upstream].as_ref().unwrap() {
                        *common_count.entry(*a).or_default() += 1;
                    }
                }
                let out = common_count
                    .iter()
                    .filter(|(_g, c)| **c == g.upstream.len())
                    .map(|(g, _c)| g)
                    .max_by_key(|g| ranks[**g]);
                assert!(out.is_some());
                out.cloned()
            })
            .collect();

        let mut children: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
            IndexVec::new();
        children.resize_with(t.groups.len(), BTreeSet::new);
        for (g, p) in parents.enumerate() {
            if let Some(p) = *p {
                children[p].insert(g);
            }
        }

        let new_groups = children
            .into_iter()
            .map(|c| c.into_iter().collect()) // BTreeSet -> Vec
            .zip(parents.into_iter())
            .zip(t.groups.iter())
            .map(|((children, parent), group)| Group {
                choices: group.choices.clone(),
                nodes: group.nodes.clone(),
                upstream: group.upstream.clone(),
                downstream: group.downstream.clone(),
                parent,
                children,
            })
            .collect();

        Stage3 {
            ops: t.ops.clone(),
            root: t.root,
            groups: new_groups,
            num_choices: t.num_choices,
            vars: t.vars.clone(),
        }
    }
}

fn populate_ancestors(
    t: &Stage2,
    g: GroupIndex,
    out: &mut IndexVec<Option<BTreeSet<GroupIndex>>, GroupIndex>,
) {
    if out[g].is_some() {
        return;
    }
    let mut s = BTreeSet::new();
    s.insert(g); // Groups are in their own ancestor list
    for upstream_group in &t.groups[g].upstream {
        // Recurse upwards towards the root
        populate_ancestors(t, *upstream_group, out);

        // XXX this is O(N^2) in memory, because nodes deeper in the tree
        // accumulate all of the ancestors of their parents
        s.extend(out[*upstream_group].as_ref().unwrap().iter().cloned());
    }
    out[g] = Some(s);
}

fn populate_ranks(
    t: &Stage2,
    g: GroupIndex,
    rank: usize,
    out: &mut IndexVec<Option<usize>, GroupIndex>,
) {
    if let Some(r) = out[g] {
        // Nothing to do here, other than a sanity-check
        assert!(r <= rank);
    } else {
        for downstream_group in &t.groups[g].downstream {
            populate_ranks(t, *downstream_group, rank + 1, out);
        }
        out[g] = Some(rank);
    }
}
