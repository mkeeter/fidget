use crate::{
    context::Context,
    error::Error,
    indexed::IndexMap,
    op::{Node, Op, VarNode},
    program::{Block, Choice, ChoiceIndex, Instruction, RegIndex, VarIndex},
};

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

/// Represents a `min` or `max` node that may directly activate a child node
///
/// (i.e. without any other `min` / `max` nodes in the way)
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum Source {
    Root,
    Left(Node),
    Right(Node),
    Both(Node),
}

impl Source {
    /// Returns the [Node] if present; otherwise returns `None`
    fn node(&self) -> Option<Node> {
        match self {
            Source::Left(a) | Source::Right(a) | Source::Both(a) => Some(*a),
            Source::Root => None,
        }
    }
}

/// Various node ranks, grouped for convenience
pub struct Rank {
    /// Minimum distance to the root (the root has value 0)
    root_min: usize,
    /// Maximum distance to the root (the root has value 0)
    root_max: usize,
    /// Minimum distance to a leaf (constants and vars have value 0)
    leaf_min: usize,
    /// Maximum distance to a leaf (constants and vars have value 0)
    leaf_max: usize,
}

impl Default for Rank {
    fn default() -> Self {
        Rank {
            root_min: usize::MAX,
            root_max: 0,
            leaf_min: usize::MAX,
            leaf_max: 0,
        }
    }
}

/// Groups are uniquely identified by a set of `Source` nodes
type GroupId = BTreeSet<Source>;

/// Represents a [Context] along with meta-data to efficiently generate code
/// for the given tree.
pub struct Compiler<'a> {
    pub ctx: &'a Context,
    pub root: Node,
    parent: BTreeMap<Node, GroupId>,
    groups: BTreeMap<GroupId, Vec<Node>>,

    /// The parent group of the given group.
    tree: BTreeMap<GroupId, Vec<GroupId>>,

    /// Node ranks of various flavors
    rank: BTreeMap<Node, Rank>,

    /// Group ranks, which are `min(root_max)` within the group
    group_rank: BTreeMap<GroupId, usize>,
}

impl<'a> Compiler<'a> {
    pub fn new(ctx: &'a Context, root: Node) -> Self {
        let mut out = Self {
            ctx,
            root,
            parent: Default::default(),
            groups: Default::default(),
            tree: Default::default(),
            rank: Default::default(),
            group_rank: Default::default(),
        };
        out.initialize();
        out
    }

    fn initialize(&mut self) {
        self.find_ranks(self.root, 0);
        self.find_groups(self.root, Source::Root);

        // Any group which includes the root will _always_ be evaluated, so
        // reset it to just contain `Source::Root`
        for g in self.parent.values_mut() {
            if g.contains(&Source::Root) {
                *g = [Source::Root].into_iter().collect();
            }
        }

        // Flip the Node -> Group map into Group -> Vec<Node>
        for (node, source) in self.parent.iter() {
            self.groups.entry(source.clone()).or_default().push(*node);
        }

        // Build the hierarchical GroupId tree
        for g in self.groups.keys() {
            let parents: BTreeSet<Node> =
                g.iter().filter_map(|s| s.node()).collect();
            if let Some(a) = self.least_common_ancestor(&parents) {
                self.tree
                    .entry(self.parent.get(&a).unwrap().clone())
                    .or_default()
                    .push(g.clone());
            }
        }

        // Recursively collect the min rank for each (hierarchical) group
        // We accumulate into a separate map because otherwise borrowing
        // doesn't quite work (since we need to borrow self.group_ranks
        // mutably, but also borrow other parts of self immutably).
        let mut group_ranks = BTreeMap::new();
        self.find_group_ranks(
            &[Source::Root].into_iter().collect(),
            &mut group_ranks,
        );
        self.group_rank = group_ranks;
    }

    /// Recursively finds group ranks, populating the provided map and
    /// returning the rank of the given group.
    fn find_group_ranks(
        &self,
        s: &GroupId,
        group_ranks: &mut BTreeMap<GroupId, usize>,
    ) -> usize {
        let mut min_rank = usize::MAX;
        if let Some(nodes) = self.groups.get(s) {
            let node_min_rank = nodes
                .iter()
                .map(|n| self.rank.get(n).unwrap().root_max)
                .min()
                .unwrap_or(usize::MAX);
            min_rank = min_rank.min(node_min_rank);
        }
        if let Some(subtrees) = self.tree.get(s) {
            for t in subtrees.iter() {
                min_rank = min_rank.min(self.find_group_ranks(t, group_ranks));
            }
        }
        group_ranks.insert(s.clone(), min_rank);
        min_rank
    }

    pub fn to_tape(
        &self,
        regs: &mut IndexMap<Node, RegIndex>,
        vars: &mut IndexMap<VarNode, VarIndex>,
        choices: &mut IndexMap<Node, ChoiceIndex>,
    ) -> Vec<Instruction> {
        let start = [Source::Root].into_iter().collect();
        self.to_tape_inner(&start, regs, vars, choices)
    }
    fn to_tape_inner(
        &self,
        s: &GroupId,
        regs: &mut IndexMap<Node, RegIndex>,
        vars: &mut IndexMap<VarNode, VarIndex>,
        choices: &mut IndexMap<Node, ChoiceIndex>,
    ) -> Vec<Instruction> {
        #[derive(Debug)]
        enum Target<'a> {
            Node(Node),
            Group(&'a GroupId),
        }
        let mut targets: Vec<Target> = self
            .groups
            .get(s)
            .into_iter()
            .flat_map(|i| i.iter().cloned().map(Target::Node))
            .chain(
                self.tree
                    .get(s)
                    .into_iter()
                    .flat_map(|t| t.iter().map(Target::Group)),
            )
            .collect();
        targets.sort_by_key(|t| match t {
            Target::Node(n) => self.rank.get(n).unwrap().root_max,
            Target::Group(g) => *self.group_rank.get(g).unwrap(),
        });
        let mut out = vec![];
        for t in targets.iter().rev() {
            match t {
                Target::Node(n) => out.push(
                    self.ctx
                        .get_op(*n)
                        .unwrap()
                        .to_instruction(*n, regs, vars, choices),
                ),
                Target::Group(g) => {
                    // If there are Both sources here, then we *must* evaluate
                    // this group (because we're bound to take one or the
                    // other branch).
                    // XXX Is this right??
                    if g.iter().any(|g| matches!(g, Source::Both(..))) {
                        out.extend(self.to_tape_inner(g, regs, vars, choices));
                    } else {
                        out.push(Instruction::Cond(
                            g.iter()
                                .map(|s| match s {
                                    Source::Left(n) => {
                                        (choices.insert(*n), Choice::Left)
                                    }
                                    Source::Right(n) => {
                                        (choices.insert(*n), Choice::Right)
                                    }
                                    _ => panic!(),
                                })
                                .collect(),
                            Block(self.to_tape_inner(g, regs, vars, choices)),
                        ))
                    }
                }
            }
        }
        out
    }

    pub fn write_dot_grouped<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        writeln!(w, "digraph mygraph {{")?;
        writeln!(w, "compound=true")?;
        self.write_dot_grouped_inner(
            w,
            &[Source::Root].into_iter().collect(),
            &mut 0,
        )?;
        // Write edges afterwards, after all nodes have been defined
        for node in self.groups.values().flat_map(|g| g.iter()) {
            self.ctx.write_edges_dot(w, *node)?;
        }
        writeln!(w, "}}")?;
        Ok(())
    }
    fn write_dot_grouped_inner<W: Write>(
        &self,
        w: &mut W,
        s: &GroupId,
        subgraph_num: &mut usize,
    ) -> Result<(), Error> {
        let groups = self.groups.get(s).unwrap();
        let subtrees = self.tree.get(s);

        let with_subgraph = groups.len() > 1 || subtrees.is_some();
        if with_subgraph {
            write!(w, "subgraph cluster_{} {{", subgraph_num)?;
        }

        *subgraph_num += 1;

        for node in groups {
            self.ctx.write_node_dot(w, *node)?;
        }
        for t in subtrees.iter().flat_map(|t| t.iter()) {
            self.write_dot_grouped_inner(w, t, subgraph_num)?;
        }
        if with_subgraph {
            write!(w, "}}")?;
        }
        Ok(())
    }

    fn find_ranks(&mut self, node: Node, root_rank: usize) {
        let e = self.rank.entry(node).or_default();
        e.root_max = e.root_max.max(root_rank);
        e.root_min = e.root_min.min(root_rank);

        let op = self.ctx.get_op(node).unwrap();
        for i in op.iter_children() {
            self.find_ranks(i, root_rank + 1);
        }

        let (leaf_min, leaf_max) = if matches!(op, Op::Const(..) | Op::Var(..))
        {
            (0, 0)
        } else {
            let leaf_min = op
                .iter_children()
                .map(|i| self.rank.get(&i).unwrap().leaf_min)
                .min()
                .unwrap();
            let leaf_max = op
                .iter_children()
                .map(|i| self.rank.get(&i).unwrap().leaf_max)
                .max()
                .unwrap();
            (leaf_min + 1, leaf_max + 1)
        };
        let e = self.rank.entry(node).or_default();
        e.leaf_min = leaf_min;
        e.leaf_max = leaf_max;
    }

    fn find_groups(&mut self, node: Node, source: Source) {
        // Update this node's parents
        let c = self.parent.entry(node).or_default();
        match source {
            Source::Left(n) if c.contains(&Source::Right(n)) => {
                c.remove(&Source::Right(n));
                c.insert(Source::Both(n));
            }
            Source::Right(n) if c.contains(&Source::Left(n)) => {
                c.remove(&Source::Left(n));
                c.insert(Source::Both(n));
            }
            Source::Left(n) | Source::Right(n)
                if c.contains(&Source::Both(n)) =>
            {
                // Nothing to do here
            }
            Source::Root | Source::Left(..) | Source::Right(..) => {
                c.insert(source);
            }
            Source::Both(..) => panic!("source should never be `Both`"),
        };

        // Recurse!
        let op = self.ctx.get_op(node).unwrap();
        match op {
            // If this node is a min/max node, then it becomes the source of
            // child nodes.
            Op::Min(a, b) | Op::Max(a, b) => {
                self.find_groups(*a, Source::Left(node));
                self.find_groups(*b, Source::Right(node));
            }
            _ => op.iter_children().for_each(|c| self.find_groups(c, source)),
        }
    }

    pub fn write_dot<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        writeln!(w, "digraph mygraph {{")?;
        writeln!(w, "compound=true")?;
        for (i, group) in self.groups.values().enumerate() {
            writeln!(w, "subgraph cluster_{} {{", i)?;
            for node in group {
                self.ctx.write_node_dot(w, *node)?;
            }
            writeln!(w, "}}")?;
        }
        // Write edges afterwards, after all nodes have been defined
        for node in self.groups.values().flat_map(|g| g.iter()) {
            self.ctx.write_edges_dot(w, *node)?;
        }
        writeln!(w, "}}")?;
        Ok(())
    }

    /*
    /// Calculate the common ancestor that's farthest from the root
    ///
    /// Returns `None` if there is no such ancestor (i.e. one of the inputs
    /// is the root of the graph)
    fn least_common_ancestor(
        &self,
        parents: &BTreeSet<Source>,
    ) -> Option<Node> {
        // We need two data structures to efficiently search for least
        // common ancestor (which is defined as the common ancestor with the
        // highest rank, i.e. farthest from the root of the tree).
        //
        // The first is a set of (max-rank, [(rank, node)]) groups, which
        // are always kept sorted, e.g.
        // [(3, [(2, a), (3, b))], (5, [(5, c)], (5, [(2, a), (4, d), (5, e)])]
        // This duplicates data in `self.rank`, but simplifies implementation.
        //
        // The second is a set of active nodes and how many groups they appear
        // in right now:
        //  {a: 2, b: 1, c: 1, d: 1, e: 1}
        //
        // If any node appears in all the groups, then we're done and can
        // return it right away.
        //
        // Now, the algorithm is as follows:
        //  - Select the group with the overall max-rank node
        //  - Remove that max-rank node from the active nodes set
        //  - Remove that max-rank node from the group
        //      - Within that group, replace the max-rank node with all of
        //        its parents (and their appropriate ranks)
        //      - For each parent, if it wasn't already in the group,
        //        increment the count in the active nodes set.  If the count
        //        hits the group count, return that node.
        //  - Reinsert the group, with a new max rank (possibly)

        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        struct GroupHandle {
            rank: usize,
            index: usize,
        }

        let mut groups = vec![];
        let mut wavefront: BTreeSet<GroupHandle> = BTreeSet::new();
        let mut active: BTreeMap<Node, usize> = BTreeMap::new();
        for (i, n) in parents.iter().enumerate() {
            match n.node() {
                Some(n) => {
                    let rank = self.rank.get(&n).unwrap().root_max;
                    // Using a BTreeSet as a de-facto priority queue. Normally, you'd
                    // be tempted to reach for BinaryHeap, but that doesn't provide
                    // deduplication.
                    let heap: BTreeSet<(usize, Node)> =
                        [(rank, n)].into_iter().collect();

                    let e = active.entry(n).or_default();
                    *e += 1;
                    if *e == parents.len() {
                        return Some(n);
                    }
                    wavefront.insert(GroupHandle { rank, index: i });
                    groups.push(heap);
                }
                // If there is a Root among the parents, then the least common
                // ancestor is by definition the root cluster.
                None => return None,
            }
        }

        loop {
            // Awkwardly work around the lack of `BTreeSet::pop_last`
            // (https://github.com/rust-lang/rust/issues/62924)
            let group = *wavefront.iter().last().unwrap();
            wavefront.remove(&group);

            let (node_rank, node) = *groups[group.index].iter().last().unwrap();
            groups[group.index].remove(&(node_rank, node));

            // This node is being removed from its group
            *active.get_mut(&node).unwrap() -= 1;

            let sources = self.parent.get(&node).unwrap();
            for n in sources {
                if let Some(n) = n.node() {
                    let rank = self.rank.get(&n).unwrap().root_max;

                    // If this ndoe wasn't already in the group, then increment
                    // the active count and see if it's now in every group
                    if groups[group.index].insert((rank, n)) {
                        let e = active.entry(n).or_default();
                        *e += 1;
                        if *e == parents.len() {
                            return Some(n);
                        }
                    }
                }
            }
            // Reinsert this group
            let new_rank = groups[group.index].iter().last().unwrap().0;
            wavefront.insert(GroupHandle {
                rank: new_rank,
                index: group.index,
            });
        }
    }
    */

    /// Finds every node which is an ancestor (closer to the root) of the
    /// given node, including itself.
    fn ancestors(&self, node: Node) -> BTreeSet<Node> {
        let mut out = BTreeSet::new();
        self.ancestors_inner(node, &mut out);
        out
    }
    fn ancestors_inner(&self, node: Node, out: &mut BTreeSet<Node>) {
        out.insert(node);
        for source in self.parent.get(&node).unwrap() {
            if let Some(n) = source.node() {
                self.ancestors_inner(n, out);
            }
        }
    }
    fn common_ancestors(&self, node: Node) -> BTreeSet<Node> {
        let mut iter = self
            .parent
            .get(&node)
            .unwrap()
            .iter()
            .filter_map(|i| i.node())
            .map(|i| self.common_ancestors(i));

        let mut out = match iter.next() {
            None => return [node].into_iter().collect(),
            Some(e) => e,
        };
        for rest in iter {
            out = out.intersection(&rest).cloned().collect();
        }
        out.insert(node);
        out
    }
    /// Calculate the common ancestor that's farthest from the root
    ///
    /// TODO: this is very inefficient. A wavefront-style search would be
    /// much faster, since it wouldn't need to find every ancestor of every
    /// node.
    fn least_common_ancestor(&self, nodes: &BTreeSet<Node>) -> Option<Node> {
        let mut iter = nodes.iter().map(|n| self.common_ancestors(*n));
        let mut out = iter.next()?;
        for rest in iter {
            out = out.intersection(&rest).cloned().collect();
        }
        out.iter()
            .max_by_key(|n| self.rank.get(n).unwrap().root_max)
            .cloned()
    }
}
