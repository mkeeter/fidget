use std::collections::BTreeMap;

use crate::{
    compiler::{ChoiceIndex, Compiler, NodeIndex, Op, VarIndex},
    context::{Context, Node, VarNode},
};

pub(crate) fn run(ctx: &Context, root: Node, out: &mut Compiler) {
    let mut seen = BTreeMap::new();
    let mut vars = BTreeMap::new();
    out.root = recurse(ctx, root, &mut seen, &mut vars, out);
}

fn recurse(
    ctx: &Context,
    node: Node,
    seen: &mut BTreeMap<Node, NodeIndex>,
    vars: &mut BTreeMap<VarNode, VarIndex>,
    out: &mut Compiler,
) -> NodeIndex {
    if let Some(i) = seen.get(&node) {
        return *i;
    }
    use crate::context::Op as CtxOp;

    let op = match ctx.get_op(node).unwrap() {
        CtxOp::Binary(op, a, b) => Op::Binary(
            *op,
            recurse(ctx, *a, seen, vars, out),
            recurse(ctx, *b, seen, vars, out),
        ),
        CtxOp::BinaryChoice(op, a, b, _) => {
            let choice_idx = ChoiceIndex::from(out.num_choices);
            out.num_choices += 1;
            Op::BinaryChoice(
                *op,
                recurse(ctx, *a, seen, vars, out),
                recurse(ctx, *b, seen, vars, out),
                choice_idx,
            )
        }
        CtxOp::Unary(op, a) => {
            Op::Unary(*op, recurse(ctx, *a, seen, vars, out))
        }
        CtxOp::Const(f) => Op::Const(f.0),
        CtxOp::Var(v) => {
            let v = vars.entry(*v).or_insert_with(|| {
                let var_name = ctx.get_var_by_index(*v).unwrap().to_owned();
                out.vars.insert(var_name)
            });
            Op::Var(*v)
        }
    };
    let idx = out.ops.push(op);
    seen.insert(node, idx);
    idx
}
