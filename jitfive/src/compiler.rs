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

/// Represents a [Context] along with meta-data to efficiently generate code
/// for the given tree.
pub struct Compiler<'a> {
    ctx: &'a Context,
    root: Node,
    parent: BTreeMap<Node, BTreeSet<Source>>,
    groups: BTreeMap<BTreeSet<Source>, Vec<Node>>,

    /// The parent group of the given group.
    tree: BTreeMap<BTreeSet<Source>, Vec<BTreeSet<Source>>>,

    /// Minimum distance to the root of the tree; the root has rank 0
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
                out.least_common_ancestor(g.iter().filter_map(|g| match g {
                    Source::Left(a) | Source::Right(a) | Source::Both(a) => {
                        Some(*a)
                    }
                    Source::Root => None,
                }));
            if let Some(a) = a {
                out.tree
                    .entry(out.parent.get(&a).unwrap().clone())
                    .or_default()
                    .push(g.clone());
            }
        }
        out.bla_recurse(&[Source::Root].into_iter().collect(), 0);
        out
    }

    fn bla_recurse(&self, s: &BTreeSet<Source>, indent: usize) {
        println!("{:indent$}{:?}", "", s);
        let indent = indent + 2;
        for n in self.groups.get(s).unwrap() {
            println!(
                "{:indent$}{:?} {:?}",
                "",
                n,
                self.ctx.get_op(*n).unwrap()
            );
        }
        for t in self.tree.get(s).iter().flat_map(|t| t.iter()) {
            self.bla_recurse(t, indent);
        }
    }

    fn find_groups(&mut self, node: Node, source: Source, rank: usize) {
        let r = self.rank.entry(node).or_insert(rank);
        *r = rank.min(*r);
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
        println!("groups: {:?}", self.groups.len());
        writeln!(w, "digraph mygraph {{")?;
        writeln!(w, "compound=true")?;
        for (i, group) in self.groups.values().enumerate() {
            writeln!(w, "subgraph cluster_{} {{", i)?;
            for node in group {
                self.ctx.write_node_dot(*node, w)?;
            }
            writeln!(w, "}}")?;
        }
        // Write edges afterwards, after all nodes have been defined
        for node in self.groups.values().flat_map(|g| g.iter()) {
            self.ctx.write_edges_dot(*node, w)?;
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
        for b in self.parent.get(&node).unwrap() {
            match b {
                Source::Left(a) | Source::Right(a) | Source::Both(a) => {
                    self.ancestors_inner(*a, out)
                }
                Source::Root => (),
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
