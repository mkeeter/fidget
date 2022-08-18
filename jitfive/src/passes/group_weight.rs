use crate::{compiler::Compiler, compiler::GroupIndex, indexed::IndexVec};

fn recurse(
    out: &Compiler,
    g: GroupIndex,
    child_weights: &mut IndexVec<Option<usize>, GroupIndex>,
) -> usize {
    let group = &out.groups[g];
    let mut sum = 0;
    for &c in &group.children {
        sum += recurse(out, c, child_weights);
    }
    child_weights[g] = Some(sum);
    sum + group.nodes.len() + group.choices.len()
}

pub(crate) fn run(out: &mut Compiler) {
    let root_group_index = out.op_group[out.root];
    let mut child_weights = IndexVec::new();
    child_weights.resize(out.groups.len(), None);
    recurse(out, root_group_index, &mut child_weights);
}
