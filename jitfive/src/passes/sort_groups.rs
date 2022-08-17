use crate::{compiler::Compiler, compiler::GroupIndex, indexed::IndexVec};

pub(crate) fn run(out: &mut Compiler) {
    let group_ranks = populate_group_ranks(out);
    for g in out.groups.iter_mut() {
        g.children
            .sort_by_key(|g| std::cmp::Reverse(group_ranks[*g].unwrap()));
    }
}

/// Populates group ranks from the perspective of the nested group tree.
///
/// Within each group, ranks are relative to the root of that group; 0 is
/// closest to the group root.
fn populate_group_ranks(t: &Compiler) -> IndexVec<Option<usize>, GroupIndex> {
    let mut ranks = IndexVec::new();
    ranks.resize(t.groups.len(), None);
    let mut out = IndexVec::new();
    out.resize(t.groups.len(), None);

    let root_group = t.op_group[t.root];
    recurse_group_ranks(t, root_group, &mut ranks, &mut out);
    out.into_iter().collect()
}

fn recurse_group_ranks(
    t: &Compiler,
    g: GroupIndex,
    ranks: &mut IndexVec<Option<usize>, GroupIndex>,
    out: &mut IndexVec<Option<usize>, GroupIndex>,
) {
    // Update this group's rank based on the current depth (stored in the parent
    // slot of the `ranks` array).
    let parent = t.groups[g].parent;
    if let Some(parent) = parent {
        let r = ranks[parent].unwrap();
        match out[g].as_mut() {
            Some(q) => *q = (*q).max(r),
            None => out[g] = Some(r),
        }
    }

    // This is tricky: we can recurse into other groups, so we need to track
    // multiple different ranks simultaneously.
    if let Some(parent) = parent {
        *ranks[parent].as_mut().unwrap() += 1;
    }
    for child in &t.groups[g].downstream {
        // Before entering a child group, set _our_ rank to 0, because that's
        // what it will look up and store.
        assert!(ranks[g].is_none());
        ranks[g] = Some(0);
        recurse_group_ranks(t, *child, ranks, out);
        assert_eq!(ranks[g], Some(0));
        ranks[g] = None;
    }
    if let Some(parent) = parent {
        *ranks[parent].as_mut().unwrap() -= 1;
    }
}
