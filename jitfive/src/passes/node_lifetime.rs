use crate::{
    compiler::GroupIndex,
    compiler::{Compiler, NodeIndex, Op},
    indexed::IndexVec,
};

pub(crate) fn run(out: &mut Compiler) {
    // Use a separate `last_use` array to avoid lifetime issues
    let mut last_use = IndexVec::new();
    last_use.resize(out.ops.len(), 0);
    let mut i = 0;
    recurse(out, out.op_group[out.root], &mut i, &mut last_use);
    std::mem::swap(&mut out.last_use, &mut last_use)
}

fn recurse(
    out: &Compiler,
    g: GroupIndex,
    i: &mut usize,
    last_use: &mut IndexVec<usize, NodeIndex>,
) {
    let group = &out.groups[g];
    for &c in &group.children {
        recurse(out, c, i, last_use);
    }
    for &n in &group.nodes {
        match out.ops[n] {
            Op::Var(..) | Op::Const(..) => (),
            Op::Binary(_op, a, b) => {
                last_use[a] = *i;
                last_use[b] = *i;
            }
            Op::BinaryChoice(_op, a, b, ..) => {
                last_use[a] = *i;
                last_use[b] = *i;
            }
            Op::Unary(_op, a) => {
                last_use[a] = *i;
            }
        }
        *i += 1;
    }
}
