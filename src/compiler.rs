use crate::{
    error::Error,
    op::{Node, Op},
    Context,
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

pub struct Compiler<'a> {
    ctx: &'a Context,
    root: Node,
    parent: BTreeMap<Node, BTreeSet<Source>>,
    groups: BTreeMap<BTreeSet<Source>, BTreeSet<Node>>,
}

impl<'a> Compiler<'a> {
    pub fn new(ctx: &'a Context, root: Node) -> Self {
        let mut parent = BTreeMap::new();
        Self::find_groups(ctx, root, Source::Root, &mut parent);

        let mut groups: BTreeMap<_, BTreeSet<Node>> = BTreeMap::new();
        for (node, source) in parent.iter() {
            groups.entry(source.clone()).or_default().insert(*node);
        }

        Self {
            ctx,
            root,
            parent,
            groups,
        }
    }

    fn find_groups(
        ctx: &Context,
        node: Node,
        parent: Source,
        data: &mut BTreeMap<Node, BTreeSet<Source>>,
    ) {
        // Update this node's parents
        let c = data.entry(node).or_default();
        match parent {
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
                c.insert(parent);
            }
            Source::Both(..) => panic!("parent should never be `Both`"),
        };

        match ctx.get_op(node).unwrap() {
            // If this node is a min/max node, then it becomes the parent of
            // child nodes.
            Op::Min(a, b) | Op::Max(a, b) => {
                Self::find_groups(ctx, *a, Source::Left(node), data);
                Self::find_groups(ctx, *b, Source::Right(node), data);
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                Self::find_groups(ctx, *a, parent, data);
                Self::find_groups(ctx, *b, parent, data);
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
            | Op::Ln(a) => Self::find_groups(ctx, *a, parent, data),

            Op::Var(..) | Op::Const(..) => (),
        }
    }

    pub fn write_dot<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        println!("groups: {:?}", self.groups.len());
        writeln!(w, "digraph mygraph {{")?;
        writeln!(w, "compound=true")?;
        let mut subgraph_num = 0;
        for group in self.groups.values() {
            if group.len() > 1 {
                writeln!(w, "subgraph cluster_{} {{", subgraph_num)?;
                subgraph_num += 1;
            }
            for node in group {
                self.ctx.write_node_dot(*node, w)?;
            }
            if group.len() > 1 {
                writeln!(w, "}}")?;
            }
        }
        for node in self.groups.values().flat_map(|g| g.iter()) {
            self.ctx.write_edges_dot(*node, w)?;
        }
        writeln!(w, "}}")?;
        Ok(())
    }
}
