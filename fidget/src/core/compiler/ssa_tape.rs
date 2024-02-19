//use crate::vm::{RegisterAllocator, Tape as VmTape};
use crate::{
    compiler::SsaOp,
    context::{BinaryOpcode, Node, Op, UnaryOpcode},
    vm::Choice,
    Context, Error,
};

use bimap::BiHashMap;
use std::collections::{HashMap, HashSet};

/// Instruction tape, storing groups of [opcodes in SSA form](crate::compiler::SsaOp)
///
/// The [`SsaRoot`] is typically stored in an `Arc`, so all of its fields are
/// public (but should be immutable in normal usage).
#[derive(Debug)]
pub struct SsaRoot {
    /// Individual groups, in reverse-evaluation order
    pub groups: Vec<SsaGroup>,

    /// Total number of (SSA) operations in all the tape groups
    pub num_ops: usize,

    /// Number of choice operations in the tape
    pub choice_count: usize,

    /// Mapping from variable names (in the original [`Context`]) to indexes in
    /// the variable array used during evaluation.
    pub vars: HashMap<String, u32>,
}

/// Individual group of [SSA operations](SsaOp)
#[derive(Debug)]
pub struct SsaGroup {
    /// Operations in this group, in reverse-evaluation order
    pub ops: Vec<SsaOp>,

    /// Offset of this group's first choice in a global choice array
    pub choice_offset: usize,

    /// Subsequent groups which are **always** enabled if this group is active
    pub enable_always: Vec<usize>,

    /// Per-group choice data, in reverse-evaluation order
    pub choices: Vec<SsaChoiceData>,
}

/// Downstream groups to enable for a given choice
#[derive(Debug)]
pub struct SsaChoiceData {
    /// Group to enable if the choice includes `Choice::Left`
    pub enable_left: usize,

    /// Group to enable if the choice includes `Choice::Right`
    pub enable_right: usize,
}

impl SsaRoot {
    /// Flattens a subtree of the graph into groups of straight-line code.
    ///
    /// This should always succeed unless the `root` is from a different
    /// `Context`, in which case `Error::BadNode` will be returned.
    pub fn new(ctx: &Context, root: Node) -> Result<Self, Error> {
        // We want to build a map from Node -> choice root -> Choice
        let node_choices = {
            let mut node_choices: HashMap<Node, HashMap<Node, Choice>> =
                HashMap::new();

            // We'll use manual recursion to avoid blowing up the stack
            struct Action {
                node: Node,
                root: Node,
                choice: Choice,
            }
            let mut todo = vec![Action {
                node: root,
                root,
                choice: Choice::None,
            }];
            while let Some(Action { node, root, choice }) = todo.pop() {
                let op = ctx.get_op(node).ok_or(Error::BadNode)?;

                // Constants are skipped because they're inlined in the SsaOp
                if matches!(op, Op::Const(..)) {
                    continue;
                }

                // If we've already seen this node + choice, then no need to
                // recurse
                let c = node_choices
                    .entry(node)
                    .or_default()
                    .entry(root)
                    .or_default();
                if *c & choice != Choice::None {
                    continue;
                }
                *c |= choice;

                match op {
                    Op::Binary(
                        BinaryOpcode::Min | BinaryOpcode::Max,
                        lhs,
                        rhs,
                    ) => {
                        // Special case: min(reg, imm) and min(imm, reg) both
                        // become MinRegImm nodes, so we swap Left and Right in
                        // that case
                        let (lhs, rhs) = if matches!(
                            ctx.get_op(*lhs).unwrap(),
                            Op::Const(..)
                        ) {
                            (rhs, lhs)
                        } else {
                            (lhs, rhs)
                        };

                        todo.push(Action {
                            node: *lhs,
                            root: node,
                            choice: Choice::Left,
                        });
                        todo.push(Action {
                            node: *rhs,
                            root: node,
                            choice: Choice::Right,
                        });
                    }
                    op => {
                        for c in op.iter_children() {
                            todo.push(Action {
                                node: c,
                                root,
                                choice,
                            });
                        }
                    }
                }
            }
            node_choices
        };

        // Special case: if the tape is only a constant, then we bail out early
        // with a specially constructed SsaRoot
        if node_choices.is_empty() {
            let c = ctx.const_value(root).unwrap().unwrap() as f32;
            return Ok(SsaRoot {
                groups: vec![SsaGroup {
                    ops: vec![SsaOp::CopyImm(0, c)],
                    choice_offset: 0,
                    choices: vec![],
                    enable_always: vec![],
                }],
                num_ops: 1,
                choice_count: 0,
                vars: HashMap::new(),
            });
        }

        #[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
        struct GroupId(usize);

        #[derive(Clone, Hash, Eq, PartialEq, Debug)]
        struct Key(Vec<(Node, Choice)>);

        // Map of cheaper key indexes (usize instead of Vec<..>)
        let mut keys: BiHashMap<Key, GroupId> = BiHashMap::new();
        let mut get_group_id = |key| {
            if let Some(id) = keys.get_by_left(&key) {
                *id
            } else {
                let next = GroupId(keys.len());
                keys.insert(key, next);
                next
            }
        };

        // Build forward and reverse mappings from nodes to groups
        //
        // (we can't use a bimap here because the group -> node map is
        // many-to-one)
        let mut group_to_nodes: HashMap<GroupId, HashSet<Node>> =
            HashMap::new();
        let mut node_to_group: HashMap<Node, GroupId> = HashMap::new();
        for (n, k) in node_choices {
            // Convert into a common key type.  There's a special-case for nodes
            // which are reachable from the root of the tree, since they will
            // always be active.
            let key = if k.get(&root) == Some(&Choice::None) {
                Key(vec![])
            } else {
                let mut v: Vec<_> = k.into_iter().collect();
                v.sort_unstable();
                Key(v)
            };
            let key_id = get_group_id(key);
            group_to_nodes.entry(key_id).or_default().insert(n);
            node_to_group.insert(n, key_id);
        }

        // Build parent-child relationships between groups
        let (ordered_groups, group_index) = {
            let mut parents: HashMap<GroupId, HashSet<GroupId>> =
                HashMap::new();
            let mut children: HashMap<GroupId, HashSet<GroupId>> =
                HashMap::new();
            for (&id, nodes) in &group_to_nodes {
                for child in nodes
                    .iter()
                    .flat_map(|n| ctx.get_op(*n).unwrap().iter_children())
                {
                    if let Some(&child_group) = node_to_group.get(&child) {
                        // Ignore dependencies within the same group
                        if child_group != id {
                            parents.entry(child_group).or_default().insert(id);
                            children.entry(id).or_default().insert(child_group);
                        }
                    }
                }
            }

            // Build an ordered list of groups based on parent-child relationships
            let root_group = node_to_group[&root];
            assert!(!parents.contains_key(&root_group));
            let mut group_index = HashMap::new();
            let mut ordered_groups = vec![];
            let mut todo = vec![root_group];
            let mut seen = HashSet::new();
            while let Some(g) = todo.pop() {
                if !parents.entry(g).or_default().is_empty() || !seen.insert(g)
                {
                    continue;
                }
                group_index.insert(g, ordered_groups.len());
                ordered_groups.push(g);
                for c in children.entry(g).or_default().iter() {
                    let r = parents.get_mut(c).unwrap().remove(&g);
                    assert!(r);
                    todo.push(*c);
                }
            }
            (ordered_groups, group_index)
        };

        // Take a brief diversion to assign a global index to every node
        #[derive(Copy, Clone)]
        enum Slot {
            Reg(u32),
            Immediate(f32),
        }
        let (mapping, var_names) = {
            let mut mapping = HashMap::new();
            let mut slot_count = 0;
            let mut var_names = HashMap::new();
            let mut todo = vec![root];
            let mut seen = HashSet::new();
            while let Some(n) = todo.pop() {
                if !seen.insert(n) {
                    continue;
                }
                let op = ctx.get_op(n).unwrap();
                let prev = match op {
                    Op::Const(c) => {
                        mapping.insert(n, Slot::Immediate(c.0 as f32))
                    }
                    _ => {
                        let i = slot_count;
                        slot_count += 1;
                        if matches!(op, Op::Var(..)) {
                            let next_var = var_names.len().try_into().unwrap();
                            var_names.insert(
                                ctx.var_name(n).unwrap().unwrap().to_string(),
                                next_var,
                            );
                        }
                        mapping.insert(n, Slot::Reg(i))
                    }
                };
                assert!(prev.is_none());
                assert!(prev.is_none());
                for child in op.iter_children() {
                    todo.push(child);
                }
            }
            (mapping, var_names)
        };

        // We're almost done, hold on!  The last step is building a global SSA
        // map (one index for each node), flattening out each individual group
        // tape, and collecting them into a single SsaRoot.
        let mut choice_count = 0;
        let mut groups_out = vec![];
        for g in ordered_groups {
            // We want to build an ordered set of nodes for this group.
            //
            // We'll start by ensuring that every node has a (global) mapping and
            // recording a set of parent-child relationships
            let mut parents: HashMap<Node, HashSet<Node>> = HashMap::new();
            let mut children: HashMap<Node, HashSet<Node>> = HashMap::new();
            for &node in &group_to_nodes[&g] {
                let op = ctx.get_op(node).unwrap();
                for child in op.iter_children() {
                    // Only handle nodes in this particular group
                    if node_to_group.get(&child) != Some(&g) {
                        continue;
                    }
                    children.entry(node).or_default().insert(child);
                    parents.entry(child).or_default().insert(node);
                }
            }
            let mut todo: Vec<Node> = group_to_nodes[&g]
                .iter()
                .cloned()
                .filter(|n| !parents.contains_key(n))
                .collect();
            let mut ordered_ops = vec![];
            let mut seen = HashSet::new();
            let mut enable_always = HashSet::new();
            let mut choices = vec![];
            while let Some(n) = todo.pop() {
                if !parents.entry(n).or_default().is_empty() || !seen.insert(n)
                {
                    continue;
                }
                let Slot::Reg(i) = mapping[&n] else {
                    // Constants are skipped, because they become immediates
                    panic!("constants should not be here");
                };
                let op = match ctx.get_op(n).unwrap() {
                    Op::Input(..) => {
                        let arg = match ctx.var_name(n).unwrap().unwrap() {
                            "X" => 0,
                            "Y" => 1,
                            "Z" => 2,
                            i => panic!("Unexpected input index: {i}"),
                        };
                        SsaOp::Input(i, arg)
                    }
                    Op::Var(..) => {
                        let v = ctx.var_name(n).unwrap().unwrap();
                        let arg = var_names[v];
                        SsaOp::Var(i, arg)
                    }
                    Op::Const(..) => {
                        unreachable!("skipped above")
                    }
                    Op::Binary(op, lhs, rhs) => {
                        type RegFn = fn(u32, u32, u32) -> SsaOp;
                        type ImmFn = fn(u32, u32, f32) -> SsaOp;
                        let f: (RegFn, ImmFn, ImmFn) = match op {
                            BinaryOpcode::Add => (
                                SsaOp::AddRegReg,
                                SsaOp::AddRegImm,
                                SsaOp::AddRegImm,
                            ),
                            BinaryOpcode::Sub => (
                                SsaOp::SubRegReg,
                                SsaOp::SubRegImm,
                                SsaOp::SubImmReg,
                            ),
                            BinaryOpcode::Mul => (
                                SsaOp::MulRegReg,
                                SsaOp::MulRegImm,
                                SsaOp::MulRegImm,
                            ),
                            BinaryOpcode::Div => (
                                SsaOp::DivRegReg,
                                SsaOp::DivRegImm,
                                SsaOp::DivImmReg,
                            ),
                            BinaryOpcode::Min => (
                                SsaOp::MinRegReg,
                                SsaOp::MinRegImm,
                                SsaOp::MinRegImm,
                            ),
                            BinaryOpcode::Max => (
                                SsaOp::MaxRegReg,
                                SsaOp::MaxRegImm,
                                SsaOp::MaxRegImm,
                            ),
                        };

                        if matches!(op, BinaryOpcode::Min | BinaryOpcode::Max) {
                            // min(imm, reg) => MinImmReg(reg, imm)
                            let (lhs, rhs) = if matches!(
                                mapping[lhs],
                                Slot::Immediate(..)
                            ) {
                                (rhs, lhs)
                            } else {
                                (lhs, rhs)
                            };
                            let enable_left = node_to_group
                                .get(lhs)
                                .map(|g| group_index[g])
                                .unwrap_or(0);
                            let enable_right = node_to_group
                                .get(rhs)
                                .map(|g| group_index[g])
                                .unwrap_or(0);
                            if enable_left == enable_right {
                                enable_always.insert(enable_left);
                            } else {
                                choices.push(SsaChoiceData {
                                    enable_left,
                                    enable_right,
                                });
                            }
                        } else {
                            for c in [lhs, rhs] {
                                if let Some(g) = node_to_group.get(c) {
                                    enable_always.insert(group_index[&g]);
                                }
                            }
                        }

                        let lhs = mapping[lhs];
                        let rhs = mapping[rhs];
                        match (lhs, rhs) {
                            (Slot::Reg(lhs), Slot::Reg(rhs)) => {
                                f.0(i, lhs, rhs)
                            }
                            (Slot::Reg(arg), Slot::Immediate(imm)) => {
                                f.1(i, arg, imm)
                            }
                            (Slot::Immediate(imm), Slot::Reg(arg)) => {
                                f.2(i, arg, imm)
                            }
                            (Slot::Immediate(..), Slot::Immediate(..)) => {
                                panic!("Cannot handle f(imm, imm)")
                            }
                        }
                    }
                    Op::Unary(op, lhs) => {
                        // The lhs node should always be associated with a group,
                        // because if it was a constant, the unary op would have
                        // been constant-folded away
                        enable_always.insert(group_index[&node_to_group[&lhs]]);
                        let lhs = match mapping[lhs] {
                            Slot::Reg(r) => r,
                            Slot::Immediate(..) => {
                                panic!("Cannot handle f(imm)")
                            }
                        };
                        let op = match op {
                            UnaryOpcode::Neg => SsaOp::NegReg,
                            UnaryOpcode::Abs => SsaOp::AbsReg,
                            UnaryOpcode::Recip => SsaOp::RecipReg,
                            UnaryOpcode::Sqrt => SsaOp::SqrtReg,
                            UnaryOpcode::Square => SsaOp::SquareReg,
                        };
                        op(i, lhs)
                    }
                };
                ordered_ops.push(op);

                // Continue processing children in this group
                for c in children.entry(n).or_default().iter() {
                    let r = parents.get_mut(c).unwrap().remove(&n);
                    assert!(r);
                    todo.push(*c);
                }
            }
            // Increment our global choice counter, which sets an offset into
            // the global choice array table during simplification
            let choice_offset = choice_count;
            choice_count += choices.len();

            let mut enable_always: Vec<_> = enable_always.into_iter().collect();
            enable_always.sort_unstable();

            // We've built this group!  Hooray!
            groups_out.push(SsaGroup {
                ops: ordered_ops,
                choices,
                choice_offset,
                enable_always,
            });
        }

        let num_ops = groups_out.iter().map(|g| g.ops.len()).sum();
        Ok(SsaRoot {
            groups: groups_out,
            choice_count,
            vars: var_names,
            num_ops,
        })
    }

    /// Returns the total number of opcodes
    ///
    /// This should always be > 0; we don't use a `NonZeroUsize` because that
    /// would be unusual for a `len()` function.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.num_ops
    }

    /// Pretty-prints the given tape to `stdout`
    pub fn pretty_print(&self) {
        for &op in self.groups.iter().rev().flat_map(|g| g.ops.iter().rev()) {
            match op {
                SsaOp::Input(out, i) => {
                    println!("${out} = INPUT {i}");
                }
                SsaOp::Var(out, i) => {
                    println!("${out} = VAR {i}");
                }
                SsaOp::NegReg(out, arg)
                | SsaOp::AbsReg(out, arg)
                | SsaOp::RecipReg(out, arg)
                | SsaOp::SqrtReg(out, arg)
                | SsaOp::CopyReg(out, arg)
                | SsaOp::SquareReg(out, arg) => {
                    let op = match op {
                        SsaOp::NegReg(..) => "NEG",
                        SsaOp::AbsReg(..) => "ABS",
                        SsaOp::RecipReg(..) => "RECIP",
                        SsaOp::SqrtReg(..) => "SQRT",
                        SsaOp::SquareReg(..) => "SQUARE",
                        SsaOp::CopyReg(..) => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${arg}");
                }

                SsaOp::AddRegReg(out, lhs, rhs)
                | SsaOp::MulRegReg(out, lhs, rhs)
                | SsaOp::DivRegReg(out, lhs, rhs)
                | SsaOp::SubRegReg(out, lhs, rhs)
                | SsaOp::MinRegReg(out, lhs, rhs)
                | SsaOp::MaxRegReg(out, lhs, rhs) => {
                    let op = match op {
                        SsaOp::AddRegReg(..) => "ADD",
                        SsaOp::MulRegReg(..) => "MUL",
                        SsaOp::DivRegReg(..) => "DIV",
                        SsaOp::SubRegReg(..) => "SUB",
                        SsaOp::MinRegReg(..) => "MIN",
                        SsaOp::MaxRegReg(..) => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                SsaOp::AddRegImm(out, arg, imm)
                | SsaOp::MulRegImm(out, arg, imm)
                | SsaOp::DivRegImm(out, arg, imm)
                | SsaOp::DivImmReg(out, arg, imm)
                | SsaOp::SubImmReg(out, arg, imm)
                | SsaOp::SubRegImm(out, arg, imm)
                | SsaOp::MinRegImm(out, arg, imm)
                | SsaOp::MaxRegImm(out, arg, imm) => {
                    let (op, swap) = match op {
                        SsaOp::AddRegImm(..) => ("ADD", false),
                        SsaOp::MulRegImm(..) => ("MUL", false),
                        SsaOp::DivImmReg(..) => ("DIV", true),
                        SsaOp::DivRegImm(..) => ("DIV", false),
                        SsaOp::SubImmReg(..) => ("SUB", true),
                        SsaOp::SubRegImm(..) => ("SUB", false),
                        SsaOp::MinRegImm(..) => ("MIN", false),
                        SsaOp::MaxRegImm(..) => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                SsaOp::CopyImm(out, imm) => {
                    println!("${out} = COPY {imm}");
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ring() {
        let mut ctx = Context::new();
        let c0 = ctx.constant(0.5);
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r = ctx.add(x2, y2).unwrap();
        let c6 = ctx.sub(r, c0).unwrap();
        let c7 = ctx.constant(0.25);
        let c8 = ctx.sub(c7, r).unwrap();
        let c9 = ctx.max(c8, c6).unwrap();

        let root = SsaRoot::new(&ctx, c9).unwrap();
        assert_eq!(root.len(), 8);
    }

    #[test]
    fn test_dupe() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let x_squared = ctx.mul(x, x).unwrap();

        let root = SsaRoot::new(&ctx, x_squared).unwrap();
        assert_eq!(root.len(), 2);
    }
}
