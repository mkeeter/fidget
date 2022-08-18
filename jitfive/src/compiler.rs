use crate::{
    context::{Context, Node},
    indexed::{define_index, IndexMap, IndexVec},
    op::GenericOp,
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
define_index!(
    GroupIndex,
    "Index of a group, globally unique in the compiler pipeline"
);

/// Represents a `min` or `max` node that may directly activate a child node
///
/// (i.e. without any other `min` / `max` nodes in the way)
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum Source {
    /// This group is accessible from the root of the tree
    Root,
    /// This group is accessible if the choice node picks the LHS
    Left(ChoiceIndex),
    /// This group is accessible if the choice node picks the RHS
    Right(ChoiceIndex),
    /// This group is accessible if the choice node picks either side
    Both(ChoiceIndex),
}

pub type Op = GenericOp<VarIndex, f64, NodeIndex, ChoiceIndex>;

#[derive(Default)]
pub struct Compiler {
    /// Unordered list of operations.
    ///
    /// Created in the initial `flatten_tree` pass
    pub ops: IndexVec<Op, NodeIndex>,

    /// Root operation in the tree
    ///
    /// Created in the initial `flatten_tree` pass
    pub root: NodeIndex,

    /// Number of nodes in the tree which make LHS/RHS choices
    ///
    /// Created in the initial `flatten_tree` pass
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    ///
    /// Created in the initial `flatten_tree` pass
    pub vars: IndexMap<String, VarIndex>,

    // --------------- Stage 1 ----------------
    /// Parent group of each operation
    ///
    /// Created in the `find_groups` pass
    pub op_group: IndexVec<GroupIndex, NodeIndex>,

    /// Unordered storage of groups
    ///
    /// Created in the `find_groups` pass
    pub groups: IndexVec<Group, GroupIndex>,

    // --------------- Stage 5 ----------------
    /// Stores the last time when a node is used.
    ///
    /// The `usize` is based on an in-order traversal, iterating over groups
    /// (recursively) then nodes within a group in sorted order.
    ///
    /// Created in the `node_lifetime` pass
    pub last_use: IndexVec<usize, NodeIndex>,
}

#[derive(Default)]
pub struct Group {
    // --------------- Stage 1 ----------------
    /// Choices which enable this group of nodes.
    ///
    /// If any choice in this array is valid, then the nodes of the group are
    /// enabled.  Choices are expressed in the positive form ("if choice _i_
    /// is *Left*, then the group is enabled").
    ///
    /// This array is expected to be sorted and unique, since it is used
    /// as a key when collecting nodes into groups.
    ///
    /// Created in the `find_groups` pass
    pub choices: Vec<Source>,

    /// Nodes in this group, in arbitrary order
    ///
    /// Indexes refer to nodes in the parent stage's `ops` array
    pub nodes: Vec<NodeIndex>,

    // --------------- Stage 2 ----------------
    /// Downstream groups are farther from the root of the tree
    ///
    /// Created in the `group_graph` pass
    pub downstream: Vec<GroupIndex>,

    /// Upstream groups are closer to the root of the tree
    ///
    /// Created in the `group_graph` pass
    pub upstream: Vec<GroupIndex>,

    // --------------- Stage 3 ----------------
    /// Parent of this group in the group tree, based on lowest common ancestor
    /// of the upstream nodes.
    ///
    /// Created in the `group_tree` pass
    pub parent: Option<GroupIndex>,

    /// Children of this group in the group tree; the opposite of `parent`
    ///
    /// Created in the `group_tree` pass
    pub children: Vec<GroupIndex>,
    // --------------- Stage 4 ----------------
    // Sorts `nodes` and `children` but adds no new data
    // These are the `sort_groups` and `sort_nodes` passes
}

impl Compiler {
    pub fn new(ctx: &Context, node: Node) -> Self {
        let mut out = Self::default();
        crate::passes::flatten_tree::run(ctx, node, &mut out);
        crate::passes::find_groups::run(&mut out);
        crate::passes::group_graph::run(&mut out);
        crate::passes::group_tree::run(&mut out);
        crate::passes::sort_nodes::run(&mut out);
        crate::passes::sort_groups::run(&mut out);
        crate::passes::node_lifetime::run(&mut out);
        out
    }

    pub fn stage0_dot(&self) -> String {
        let mut out = "digraph mygraph {{\n".to_owned();
        for (i, op) in self.ops.enumerate() {
            out += &op.dot_node(i, &self.vars);
        }
        // Write edges afterwards, after all nodes have been defined
        for (i, op) in self.ops.enumerate() {
            out += &op.dot_edges(i);
        }
        out += "}}\n";
        out
    }

    pub fn stage1_dot(&self) -> String {
        let mut out = "digraph mygraph {{\ncompound=true\n".to_owned();
        for (i, group) in self.groups.enumerate() {
            out += &format!("subgraph cluster_{} {{\n", usize::from(i));
            for n in &group.nodes {
                let op = self.ops[*n];
                out += &op.dot_node(*n, &self.vars);
                out += "\n";
            }
            out += "}}\n";
        }
        // Write edges afterwards, after all nodes have been defined
        for (i, op) in self.ops.enumerate() {
            out += &op.dot_edges(i);
            out += "\n";
        }
        out += "}}\n";
        out
    }
}
