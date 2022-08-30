use std::collections::{BTreeMap, BTreeSet};

use crate::{
    compiler::{Compiler, Group, GroupIndex, NodeIndex, Op, Source},
    util::indexed::IndexVec,
};

/// Recursively collects per-node sources into the `node_sources` array
fn recurse(
    t: &Compiler,
    node: NodeIndex,
    source: Source,
    node_sources: &mut IndexVec<BTreeSet<Source>, NodeIndex>,
) {
    // Update the source value
    node_sources[node].insert(source);
    match &t.ops[node] {
        // If this node is a min/max node, then it becomes the source of
        // child nodes.
        Op::BinaryChoice(_, a, b, c) => {
            recurse(t, *a, Source::Left(*c), node_sources);
            recurse(t, *b, Source::Right(*c), node_sources);
        }
        op => op
            .iter_children()
            .for_each(|c| recurse(t, c, source, node_sources)),
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

pub(crate) fn run(out: &mut Compiler) {
    let mut sources = IndexVec::new();
    sources.resize_with(out.ops.len(), BTreeSet::new);

    recurse(out, out.root, Source::Root, &mut sources);

    // Collect node assignments into a per-group map
    let mut groups: BTreeMap<Vec<Source>, Vec<NodeIndex>> = Default::default();
    for (node_index, group_set) in sources.enumerate() {
        groups
            .entry(flatten(group_set))
            .or_default()
            .push(node_index);
    }

    // Scatter group assignments into a per-node array
    let mut gs: IndexVec<Option<GroupIndex>, NodeIndex> =
        vec![None; out.ops.len()].into();
    for (group_index, group) in groups.values().enumerate() {
        for node in group {
            let v = &mut gs[*node];
            assert_eq!(*v, None);
            *v = Some(GroupIndex::from(group_index));
        }
    }
    out.op_group = gs.into_iter().map(Option::unwrap).collect();

    out.groups = groups
        .into_iter()
        .map(|(choices, nodes)| Group {
            choices,
            nodes,
            ..Default::default()
        })
        .collect::<IndexVec<Group, GroupIndex>>();
}
