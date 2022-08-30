use crate::{
    compiler::GroupIndex,
    compiler::{Compiler, NodeIndex},
    util::indexed::IndexVec,
};

/// Populates `out[node]` with the leaf rank of the given node within its group.
///
/// A leaf rank of 0 means that the node has no children, or has exclusively
/// children from outside of the group.
///
/// Otherwise, the leaf rank is the maximum leaf rank of children plus one.
fn populate_node_ranks(t: &Compiler) -> IndexVec<usize, NodeIndex> {
    let mut out = IndexVec::new();
    out.resize(t.ops.len(), None);
    for (g, group) in t.groups.enumerate() {
        for n in &group.nodes {
            recurse_node_ranks(t, g, *n, &mut out);
        }
    }
    out.into_iter().map(Option::unwrap).collect()
}

fn recurse_node_ranks(
    t: &Compiler,
    g: GroupIndex,
    node: NodeIndex,
    out: &mut IndexVec<Option<usize>, NodeIndex>,
) -> usize {
    assert_eq!(t.op_group[node], g);
    if let Some(r) = out[node] {
        return r;
    }
    let rank = t.ops[node]
        .iter_children()
        .filter(|c| t.op_group[*c] == g)
        .map(|c| recurse_node_ranks(t, g, c, out))
        .max()
        .map(|r| r + 1)
        .unwrap_or(0);
    out[node] = Some(rank);
    rank
}

pub(crate) fn run(out: &mut Compiler) {
    let node_ranks = populate_node_ranks(out);
    for g in out.groups.iter_mut() {
        g.nodes.sort_by_key(|n| node_ranks[*n]);
    }
}
