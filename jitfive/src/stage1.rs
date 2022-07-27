use std::collections::{BTreeMap, BTreeSet};

use crate::indexed::{define_index, IndexMap, IndexVec};
use crate::stage0::{ChoiceIndex, NodeIndex, Op, Stage0, VarIndex};

define_index!(
    GroupIndex,
    "Index of a group, globally unique in the compiler pipeline"
);

/// Represents a `min` or `max` node that may directly activate a child node
///
/// (i.e. without any other `min` / `max` nodes in the way)
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum Source {
    /// This group is accessible from the root of the tree
    Root,
    /// This group is accessible if the choice node picks the LHS
    Left(ChoiceIndex),
    /// This group is accessible if the choice node picks the RHS
    Right(ChoiceIndex),
    /// This group is accessible if the choice node picks either side
    Both(ChoiceIndex),
}

/// A group represents a set of nodes which are enabled by the same set of
/// choices at `min` or `max` nodes.
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
}

/// Stores a graph of math expressions and node groups
#[derive(Debug)]
pub struct Stage1 {
    /// Math operations, stored in arbitrary order and associated with a group
    pub ops: IndexVec<(Op, GroupIndex), NodeIndex>,

    /// Root of the tree
    pub root: NodeIndex,

    /// Groups of nodes, stored in arbitrary order
    pub groups: IndexVec<Group, GroupIndex>,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,
}

/// Recursively collects per-node sources into the `out` array
fn recurse(
    t: &Stage0,
    node: NodeIndex,
    source: Source,
    out: &mut IndexVec<BTreeSet<Source>, NodeIndex>,
) {
    // Update the source value
    out[node].insert(source);
    match &t.ops[node] {
        // If this node is a min/max node, then it becomes the source of
        // child nodes.
        Op::BinaryChoice(_, a, b, c) => {
            recurse(t, *a, Source::Left(*c), out);
            recurse(t, *b, Source::Right(*c), out);
        }
        op => op.iter_children().for_each(|c| recurse(t, c, source, out)),
    }
}

/// Converts a source set into a flat sorted `Vec`, suitable for use as a group
/// key.
///
/// This process merges `Source::Left(n) + Source::Right(n) => Source::Both(n)`,
/// and drops all other sources if `Source::Root` is found.
fn flatten(input: &BTreeSet<Source>) -> Vec<Source> {
    if input.contains(&Source::Root) {
        return vec![Source::Root];
    }
    let mut out = vec![];
    for i in input {
        match i {
            Source::Left(n) if input.contains(&Source::Right(*n)) => {
                out.push(Source::Both(*n))
            }
            Source::Right(n) if input.contains(&Source::Left(*n)) => {
                // Do nothing; it's pushed in the Left branch above
            }
            // Simplest case
            Source::Left(..) | Source::Right(..) => out.push(*i),

            Source::Root => unreachable!("`Source::Root` check failed?"),
            Source::Both(..) => panic!("Should not have `Both` here!"),
        }
    }
    out.sort();
    out
}

impl From<&Stage0> for Stage1 {
    fn from(t: &Stage0) -> Self {
        let mut sources = IndexVec::new();
        sources.resize_with(t.ops.len(), BTreeSet::new);

        recurse(t, t.root, Source::Root, &mut sources);

        // Collect node assignments into a per-group map
        let mut groups: BTreeMap<Vec<Source>, Vec<NodeIndex>> =
            Default::default();
        for (node_index, group_set) in sources.enumerate() {
            groups
                .entry(flatten(group_set))
                .or_default()
                .push(node_index);
        }

        // Scatter group assignments into a per-node array
        let mut gs: IndexVec<Option<GroupIndex>, NodeIndex> =
            vec![None; t.ops.len()].into();
        for (group_index, group) in groups.values().enumerate() {
            for node in group {
                let v = &mut gs[*node];
                assert_eq!(*v, None);
                *v = Some(GroupIndex::from(group_index));
            }
        }
        let ops = t
            .ops
            .iter()
            .cloned()
            .zip(gs.into_iter().map(Option::unwrap))
            .collect::<IndexVec<(Op, GroupIndex), NodeIndex>>();

        let groups = groups
            .into_iter()
            .map(|(choices, nodes)| Group { choices, nodes })
            .collect::<IndexVec<Group, GroupIndex>>();

        Stage1 {
            ops,
            groups,
            root: t.root,
            num_choices: t.num_choices,
            vars: t.vars.clone(),
        }
    }
}
