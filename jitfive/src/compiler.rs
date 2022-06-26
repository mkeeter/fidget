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
    groups: BTreeMap<BTreeSet<Source>, BTreeSet<Node>>,

    /// Minimum distance to the root of the tree; the root has rank 0
    rank: BTreeMap<Node, usize>,
}

impl<'a> Compiler<'a> {
    pub fn new(ctx: &'a Context, root: Node) -> Self {
        let mut parent = BTreeMap::new();
        let mut rank = BTreeMap::new();
        Self::find_groups(ctx, root, Source::Root, 0, &mut parent, &mut rank);

        let mut groups: BTreeMap<_, BTreeSet<Node>> = BTreeMap::new();
        for (node, source) in parent.iter() {
            groups.entry(source.clone()).or_default().insert(*node);
        }

        Self {
            ctx,
            root,
            parent,
            groups,
            rank,
        }
    }

    fn find_groups(
        ctx: &Context,
        node: Node,
        source: Source,
        rank: usize,
        parents: &mut BTreeMap<Node, BTreeSet<Source>>,
        ranks: &mut BTreeMap<Node, usize>,
    ) {
        let r = ranks.entry(node).or_insert(rank);
        *r = rank.min(*r);
        let rank = rank + 1;

        // Update this node's parents
        let c = parents.entry(node).or_default();
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
        match ctx.get_op(node).unwrap() {
            // If this node is a min/max node, then it becomes the source of
            // child nodes.
            Op::Min(a, b) | Op::Max(a, b) => {
                let lsource = Source::Left(node);
                Self::find_groups(ctx, *a, lsource, rank, parents, ranks);
                let rsource = Source::Right(node);
                Self::find_groups(ctx, *b, rsource, rank, parents, ranks);
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                Self::find_groups(ctx, *a, source, rank, parents, ranks);
                Self::find_groups(ctx, *b, source, rank, parents, ranks);
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
            | Op::Ln(a) => {
                Self::find_groups(ctx, *a, source, rank, parents, ranks)
            }

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
}
