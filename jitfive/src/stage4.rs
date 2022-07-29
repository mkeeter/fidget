use crate::indexed::IndexVec;
use crate::stage1::GroupIndex;
use crate::stage3::Stage3;

/// Populates group ranks from the perspective of the nested group tree.
///
/// Within each group, ranks are relative to the root of that group; 0 is
/// closest to the group root.
fn populate_ranks(
    t: &Stage3,
    g: GroupIndex,
    ranks: &mut IndexVec<Option<usize>, GroupIndex>,
    out: &mut IndexVec<usize, GroupIndex>,
) {
    out[g] = match t.groups[g].parent {
        Some(p) => ranks[p].unwrap(),
        None => 0,
    };

    // This is tricky: we can recurse into other groups, so we need to track
    // multiple different ranks simultaneously
    *ranks[g].as_mut().unwrap() += 1;
    for g in &t.groups[g].downstream {
        // Before entering a child group, set its rank to 0
        assert!(ranks[*g].is_none());
        ranks[*g] = Some(0);
        populate_ranks(t, *g, ranks, out);
        assert_eq!(ranks[*g], Some(0));
        ranks[*g] = None;
    }
    *ranks[g].as_mut().unwrap() -= 1;
}
