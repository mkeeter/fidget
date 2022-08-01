use crate::indexed::{IndexMap, IndexVec};
use crate::op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode};
use crate::stage0::{NodeIndex, Op, VarIndex};
use crate::stage1::{GroupIndex, Source, TaggedOp};
use crate::stage3::Stage3;

#[derive(Default, Debug)]
pub struct Group {
    /// Choices which enable this group of nodes.
    ///
    /// If any choice in this array is valid, then the nodes of the group are
    /// enabled.  Choices are expressed in the positive form ("if choice _i_
    /// is *Left*, then the group is enabled").
    ///
    /// This array is expected to be sorted and unique, since it is used
    /// as a key when collecting nodes into groups.
    pub choices: Vec<Source>,

    /// Nodes in this group, in arbitrary order
    ///
    /// Indexes refer to nodes in the parent stage's `ops` array.
    ///
    /// This `Vec` is sorted for in-order execution (after all `children` are
    /// evaluated recursively).
    pub nodes: Vec<NodeIndex>,

    /// Children of this group in the group tree, sorted for in-order execution
    /// (recursively)
    pub children: Vec<GroupIndex>,
}

/// Stores a graph of math expressions, a graph of node groups, and a tree of
/// node groups (based on lowest common ancestors in the graph).
///
/// Unlike `Stage2`, `groups[i].children` and `groups[i].nodes` are
/// topologically sorted, so they can be evaluated (recursively) in order.
#[derive(Debug)]
pub struct Stage4 {
    /// Math operations, stored in arbitrary order and associated with a group
    pub ops: IndexVec<TaggedOp, NodeIndex>,

    /// Root of the tree
    pub root: NodeIndex,

    /// Groups of nodes and group graph links, stored in arbitrary order
    pub groups: IndexVec<Group, GroupIndex>,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,
}

impl From<&Stage3> for Stage4 {
    fn from(t: &Stage3) -> Self {
        let group_ranks = populate_group_ranks(t);
        let node_ranks = populate_op_ranks(t);

        let mut groups: IndexVec<Group, GroupIndex> = IndexVec::new();
        for g in t.groups.iter() {
            let mut nodes = g.nodes.clone();
            nodes.sort_by_key(|n| node_ranks[*n]);
            let mut children = g.children.clone();
            children
                .sort_by_key(|g| std::cmp::Reverse(group_ranks[*g].unwrap()));
            groups.push(Group {
                choices: g.choices.clone(),
                nodes,
                children,
            });
        }

        Stage4 {
            ops: t.ops.clone(),
            root: t.root,
            num_choices: t.num_choices,
            groups,
            vars: t.vars.clone(),
        }
    }
}

impl Stage4 {
    pub fn to_string(&self) -> String {
        self.recurse_to_string(self.ops[self.root].group)
    }
    fn recurse_to_string(&self, g: GroupIndex) -> String {
        let mut out = String::new();
        for g in &self.groups[g].children {
            // TODO: upgrade nodes with `Both` to live with their parent group
            let has_both = self.groups[*g]
                .choices
                .iter()
                .any(|c| matches!(c, Source::Both(..)));
            if !has_both {
                let multi = self.groups[*g].choices.len() > 1;
                out += "(if";
                if multi {
                    out += " (or";
                }
                for c in self.groups[*g].choices.iter() {
                    let (selector, index) = match c {
                        Source::Left(i) => ("left", i),
                        Source::Right(i) => ("right", i),
                        Source::Both(i) => ("both", i),
                        Source::Root => panic!(),
                    };
                    out += &format!(" ({} #{})", selector, usize::from(*index));
                }
                if multi {
                    out += ")";
                }
                out += ")\n";
            }
            out += &self.recurse_to_string(*g);
            if !has_both {
                out += ")\n";
            }
        }
        for n in self.groups[g].nodes.iter() {
            out += &format!("${} = ", usize::from(*n));
            out += &match self.ops[*n].op {
                Op::Binary(op, a, b) => {
                    let op = match op {
                        BinaryOpcode::Add => "add",
                        BinaryOpcode::Mul => "mul",
                    };
                    format!("({} ${} ${})", op, usize::from(a), usize::from(b))
                }
                Op::BinaryChoice(op, a, b, c) => {
                    let op = match op {
                        BinaryChoiceOpcode::Max => "min",
                        BinaryChoiceOpcode::Min => "max",
                    };
                    format!(
                        "({} ${} ${} [#{}])",
                        op,
                        usize::from(a),
                        usize::from(b),
                        usize::from(c),
                    )
                }
                Op::Const(f) => format!("{}", f),
                Op::Unary(op, n) => {
                    let op = match op {
                        UnaryOpcode::Neg => "neg",
                        UnaryOpcode::Abs => "abs",
                        UnaryOpcode::Recip => "recip",
                        UnaryOpcode::Sqrt => "sqrt",
                        UnaryOpcode::Sin => "sin",
                        UnaryOpcode::Cos => "cos",
                        UnaryOpcode::Tan => "tan",
                        UnaryOpcode::Asin => "asin",
                        UnaryOpcode::Acos => "acos",
                        UnaryOpcode::Atan => "atan",
                        UnaryOpcode::Exp => "exp",
                        UnaryOpcode::Ln => "ln",
                    };
                    format!("({} ${})", op, usize::from(n))
                }
                Op::Var(v) => {
                    format!("(var '{}')", self.vars.get_by_index(v).unwrap())
                }
            };
            out += "\n";
        }
        out
    }
}

/// Populates group ranks from the perspective of the nested group tree.
///
/// Within each group, ranks are relative to the root of that group; 0 is
/// closest to the group root.
fn populate_group_ranks(t: &Stage3) -> IndexVec<Option<usize>, GroupIndex> {
    let mut ranks = IndexVec::new();
    ranks.resize(t.groups.len(), None);
    let mut out = IndexVec::new();
    out.resize(t.groups.len(), None);

    let root_group = t.ops[t.root].group;
    recurse_group_ranks(t, root_group, &mut ranks, &mut out);
    out.into_iter().collect()
}

fn recurse_group_ranks(
    t: &Stage3,
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

/// Populates `out[node]` with the leaf rank of the given node within its group.
///
/// A leaf rank of 0 means that the node has no children, or has exclusively
/// children from outside of the group.
///
/// Otherwise, the leaf rank is the maximum leaf rank of children plus one.
fn populate_op_ranks(t: &Stage3) -> IndexVec<usize, NodeIndex> {
    let mut out = IndexVec::new();
    out.resize(t.ops.len(), None);
    for (g, group) in t.groups.enumerate() {
        for n in &group.nodes {
            recurse_op_ranks(t, g, *n, &mut out);
        }
    }
    out.into_iter().map(Option::unwrap).collect()
}

fn recurse_op_ranks(
    t: &Stage3,
    g: GroupIndex,
    node: NodeIndex,
    out: &mut IndexVec<Option<usize>, NodeIndex>,
) -> usize {
    assert_eq!(t.ops[node].group, g);
    if let Some(r) = out[node] {
        return r;
    }
    let rank = t.ops[node]
        .op
        .iter_children()
        .filter(|c| t.ops[*c].group == g)
        .map(|c| recurse_op_ranks(t, g, c, out))
        .max()
        .map(|r| r + 1)
        .unwrap_or(0);
    out[node] = Some(rank);
    rank
}
