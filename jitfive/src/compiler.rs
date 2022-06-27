use crate::{
    context::Context,
    error::Error,
    op::{Node, Op},
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Source {
    Root,
    Left(Node),
    Right(Node),
    Both(Node),
}

impl Source {
    fn node(&self) -> Option<Node> {
        match self {
            Source::Left(a) | Source::Right(a) | Source::Both(a) => Some(*a),
            Source::Root => None,
        }
    }
}

/// Represents a [Context] along with meta-data to efficiently generate code
/// for the given tree.
pub struct Compiler<'a> {
    ctx: &'a Context,
    root: Node,
    parent: BTreeMap<Node, BTreeSet<Source>>,
    groups: BTreeMap<BTreeSet<Source>, Vec<Node>>,

    /// The parent group of the given group.
    tree: BTreeMap<BTreeSet<Source>, Vec<BTreeSet<Source>>>,

    /// Maximum distance from the root of the tree; the root has rank 0
    rank: BTreeMap<Node, usize>,
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
        };
        out.find_groups(root, Source::Root, 0);

        // Any group which includes the root will _always_ be evaluated, so
        // reset it to just contain `Source::Root`
        for g in out.parent.values_mut() {
            if g.contains(&Source::Root) {
                *g = [Source::Root].into_iter().collect();
            }
        }

        // Flip the Node -> Group map into Group -> Vec<Node>
        for (node, source) in out.parent.iter() {
            out.groups.entry(source.clone()).or_default().push(*node);
        }
        for g in out.groups.keys() {
            let a =
                out.least_common_ancestor(g.iter().filter_map(Source::node));

            if let Some(a) = a {
                out.tree
                    .entry(out.parent.get(&a).unwrap().clone())
                    .or_default()
                    .push(g.clone());
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
        s: &BTreeSet<Source>,
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

    fn find_groups(&mut self, node: Node, source: Source, rank: usize) {
        let r = self.rank.entry(node).or_insert(rank);
        *r = rank.max(*r);
        let rank = rank + 1;

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
        match self.ctx.get_op(node).unwrap() {
            // If this node is a min/max node, then it becomes the source of
            // child nodes.
            Op::Min(a, b) | Op::Max(a, b) => {
                self.find_groups(*a, Source::Left(node), rank);
                self.find_groups(*b, Source::Right(node), rank);
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                self.find_groups(*a, source, rank);
                self.find_groups(*b, source, rank);
            }

            Op::Neg(a)
            | Op::Abs(a)
            | Op::Recip(a)
            | Op::Sqrt(a)
            | Op::Sin(a)
            | Op::Cos(a)
            | Op::Tan(a)
            | Op::Asin(a)
            | Op::Acos(a)
            | Op::Atan(a)
            | Op::Exp(a)
            | Op::Ln(a) => self.find_groups(*a, source, rank),

            Op::Var(..) | Op::Const(..) => (),
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

    /// Calculate the common ancestor that's farthest from the root
    ///
    /// TODO: this is very inefficient. A wavefront-style search would be
    /// much faster, since it wouldn't need to find every ancestor of every
    /// node.
    fn least_common_ancestor<I>(&self, nodes: I) -> Option<Node>
    where
        I: IntoIterator<Item = Node>,
    {
        let mut iter = nodes.into_iter().map(|n| self.ancestors(n));
        let mut out = iter.next()?;
        for rest in iter {
            out = out.intersection(&rest).cloned().collect();
        }
        out.iter().max_by_key(|n| self.rank[n]).cloned()
    }
}
