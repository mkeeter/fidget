use std::collections::BTreeSet;

use crate::{
    compiler::{Compiler, GroupIndex},
    util::indexed::IndexVec,
};

pub(crate) fn run(out: &mut Compiler) {
    let mut downstream: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
        IndexVec::new();
    downstream.resize_with(out.groups.len(), BTreeSet::new);

    let mut upstream: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
        IndexVec::new();
    upstream.resize_with(out.groups.len(), BTreeSet::new);

    // Find group inputs and outputs by noticing cases where a child node
    // is stored in a different group than its caller.
    for (group_index, group) in out.groups.enumerate() {
        for n in group.nodes.iter() {
            for c in out.ops[*n].iter_children() {
                let child_group = out.op_group[c];
                if child_group != group_index {
                    downstream[group_index].insert(child_group);
                    upstream[child_group].insert(group_index);
                }
            }
        }
    }

    for (g, (downstream, upstream)) in out
        .groups
        .iter_mut()
        .zip(downstream.into_iter().zip(upstream.into_iter()))
    {
        g.upstream = upstream.into_iter().collect();
        g.downstream = downstream.into_iter().collect();
    }
}
