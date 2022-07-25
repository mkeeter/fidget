use std::collections::{BTreeMap, BTreeSet};

use crate::{
    context::Context,
    indexed::{define_index, IndexMap, IndexVec},
    op::GenericOp,
    op::{Node, VarNode},
};

define_index!(
    VarIndex,
    "Index of a variable, globally unique in the compiler pipeline"
);
define_index!(
    NodeIndex,
    "Index of a node, globally unique in the compiler pipeline"
);
define_index!(
    ChoiceIndex,
    "Index of a choice, globally unique in the compiler pipeline"
);

pub type Op = GenericOp<VarIndex, f64, NodeIndex, ChoiceIndex>;

/// Stage 0 of the shape compiler pipeline.
///
/// Operations are tightly packed and assigned a globally unique index.
#[derive(Default)]
pub struct Stage0 {
    /// Unordered list of operations.
    pub ops: IndexVec<Op, NodeIndex>,

    /// Root operation in the tree
    pub root: NodeIndex,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,
}

impl Stage0 {
    pub fn self_check(&self) {
        assert!(usize::from(self.root) < self.ops.len());
        let mut used = BTreeSet::new();
        let mut choices = BTreeSet::new();

        // TODO: implement IntoIter on IndexVec?
        for o in self.ops.iter() {
            for a in o.iter_children() {
                used.insert(a);
                assert!(
                    usize::from(a) < self.ops.len(),
                    "Invalid child node index"
                );
            }
            match o {
                Op::Var(v) => {
                    assert!(
                        usize::from(*v) < self.vars.len(),
                        "Invalid var index"
                    );
                }
                Op::BinaryChoice(_, _, _, c) => {
                    assert!(
                        usize::from(*c) < self.num_choices,
                        "Invalid choice index"
                    );
                    assert!(choices.insert(*c), "Duplicate choice");
                }
                _ => (),
            }
        }
        assert_eq!(
            used.len(),
            self.ops.len() - 1,
            "All nodes must be used at least once"
        );
        assert!(
            used.insert(self.root),
            "The root cannot be used in the graph"
        );

        assert_eq!(
            choices.len(),
            self.num_choices,
            "Choice array is not densely packet"
        );
    }

    pub fn from_context(ctx: &Context, root: Node) -> Self {
        let mut out = Self::default();
        let mut seen = BTreeMap::new();
        let mut vars = BTreeMap::new();
        out.root = out.recurse(ctx, root, &mut seen, &mut vars);
        out
    }

    fn recurse(
        &mut self,
        ctx: &Context,
        node: Node,
        seen: &mut BTreeMap<Node, NodeIndex>,
        vars: &mut BTreeMap<VarNode, VarIndex>,
    ) -> NodeIndex {
        if let Some(i) = seen.get(&node) {
            return *i;
        }
        use crate::op::Op as CtxOp;

        let op = match ctx.get_op(node).unwrap() {
            CtxOp::Binary(op, a, b) => Op::Binary(
                *op,
                self.recurse(ctx, *a, seen, vars),
                self.recurse(ctx, *b, seen, vars),
            ),
            CtxOp::BinaryChoice(op, a, b, _) => {
                let choice_idx = ChoiceIndex::from(self.num_choices);
                self.num_choices += 1;
                Op::BinaryChoice(
                    *op,
                    self.recurse(ctx, *a, seen, vars),
                    self.recurse(ctx, *b, seen, vars),
                    choice_idx,
                )
            }
            CtxOp::Unary(op, a) => {
                Op::Unary(*op, self.recurse(ctx, *a, seen, vars))
            }
            CtxOp::Const(f) => Op::Const(f.0),
            CtxOp::Var(v) => {
                let v = vars.entry(*v).or_insert_with(|| {
                    let var_name = ctx.get_var_by_index(*v).unwrap().to_owned();
                    self.vars.insert(var_name)
                });
                Op::Var(*v)
            }
        };
        let idx = self.ops.push(op);
        seen.insert(node, idx);
        idx
    }
}
