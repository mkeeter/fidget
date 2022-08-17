use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{compiler::Compiler, compiler::GroupIndex, indexed::IndexVec};

fn populate_common_ancestors(
    t: &Compiler,
    g: GroupIndex,
    out: &mut IndexVec<Option<BTreeSet<GroupIndex>>, GroupIndex>,
) {
    // Skip if we've already handled this node
    if out[g].is_some() {
        return;
    }
    let group = &t.groups[g];
    let mut common_count: BTreeMap<_, usize> = BTreeMap::new();
    for upstream_group in &group.upstream {
        // Recurse upwards towards the root
        populate_common_ancestors(t, *upstream_group, out);
        for a in out[*upstream_group].as_ref().unwrap() {
            *common_count.entry(*a).or_default() += 1;
        }
    }

    // Common ancestors are reachable from every upstream group of the target,
    // which we check by counting how many upstream groups can reach each
    // potential common ancestor.
    let mut s: BTreeSet<GroupIndex> = common_count
        .iter()
        .filter(|(_g, c)| **c == group.upstream.len())
        .map(|(g, _c)| *g)
        .collect();

    // Every node is its own common ancestor
    s.insert(g);

    out[g] = Some(s);
}

fn populate_ranks(
    t: &Compiler,
    g: GroupIndex,
    rank: usize,
    out: &mut IndexVec<Option<usize>, GroupIndex>,
) {
    // Breadth-first search!
    let mut todo = VecDeque::new();
    todo.push_back((g, rank));
    while let Some((g, rank)) = todo.pop_front() {
        if let Some(r) = out[g] {
            // Nothing to do here, other than a sanity-check
            assert!(r <= rank);
        } else {
            for downstream_group in &t.groups[g].downstream {
                todo.push_back((*downstream_group, rank + 1));
            }
            out[g] = Some(rank);
        }
    }
}

pub(crate) fn run(out: &mut Compiler) {
    let common_ancestors: IndexVec<BTreeSet<GroupIndex>, GroupIndex> = {
        let mut common_ancestors = IndexVec::new();
        common_ancestors.resize(out.groups.len(), None);
        for g in 0..out.groups.len() {
            populate_common_ancestors(
                out,
                GroupIndex::from(g),
                &mut common_ancestors,
            );
        }
        common_ancestors.into_iter().map(Option::unwrap).collect()
    };

    let ranks: IndexVec<usize, GroupIndex> = {
        let mut ranks = IndexVec::new();
        ranks.resize(out.groups.len(), None);
        let root_group_index = out.op_group[out.root];
        populate_ranks(out, root_group_index, 0, &mut ranks);
        ranks.into_iter().map(Option::unwrap).collect()
    };

    let parents: IndexVec<Option<GroupIndex>, GroupIndex> = out
        .groups
        .enumerate()
        .map(|(i, group)| {
            // Special case for the root of the tree
            if group.upstream.is_empty() {
                return None;
            }
            let out = common_ancestors[i]
                .iter()
                .filter(|g| **g != i)
                .max_by_key(|g| ranks[**g]);
            assert!(out.is_some());
            out.cloned()
        })
        .collect();

    let mut children: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
        IndexVec::new();
    children.resize_with(out.groups.len(), BTreeSet::new);
    for (g, p) in parents.enumerate() {
        if let Some(p) = *p {
            children[p].insert(g);
        }
    }

    for ((children, parent), group) in children
        .into_iter()
        .zip(parents.into_iter())
        .zip(out.groups.iter_mut())
    {
        group.parent = parent;
        group.children = children.into_iter().collect();
    }
}
