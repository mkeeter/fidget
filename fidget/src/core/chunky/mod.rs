use crate::{
    context::{BinaryOpcode, Context, Node, Op, UnaryOpcode, VarNode},
    eval::tracing::Choice,
    Error,
};

use std::collections::{BTreeMap, BTreeSet};

/// Globally unique index for a particular choice node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ChoiceIndex(usize);

/// A single choice at a particular node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct DnfClause {
    root: ChoiceIndex,
    choice: Choice,
}

pub(crate) struct Compiler<'a> {
    ctx: &'a Context,
    choice_id: BTreeMap<Node, ChoiceIndex>,

    /// Conditional that activates a particular node
    ///
    /// This is recorded as an `OR` of multiple [`DnfClause`] values
    node_dnfs: BTreeMap<Node, BTreeSet<Option<DnfClause>>>,
}

impl<'a> Compiler<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            choice_id: BTreeMap::default(),
            node_dnfs: BTreeMap::default(),
        }
    }

    pub fn buildy(&mut self, root: Node) -> Result<(), Error> {
        self.recurse(root, None)?;

        println!("got {} choices", self.choice_id.len());
        for d in self.node_dnfs.values_mut() {
            let mut collapsible = vec![];
            for o in d.iter() {
                match o {
                    Some(DnfClause {
                        root,
                        choice: Choice::Left,
                    }) => {
                        if d.contains(&Some(DnfClause {
                            root: *root,
                            choice: Choice::Right,
                        })) {
                            collapsible.push(*root);
                        }
                    }
                    Some(..) | None => (),
                }
            }
            println!("got {} collapsible", collapsible.len());
            for root in collapsible {
                let r = d.remove(&Some(DnfClause {
                    root,
                    choice: Choice::Left,
                }));
                assert!(r);
                let r = d.remove(&Some(DnfClause {
                    root,
                    choice: Choice::Right,
                }));
                assert!(r);
                d.insert(Some(DnfClause {
                    root,
                    choice: Choice::Both,
                }));
            }
        }

        let mut dnf_nodes: BTreeMap<_, Vec<Node>> = BTreeMap::new();
        for (n, d) in &self.node_dnfs {
            println!(
                "node {n:?} {:?} has {} DNFs",
                self.ctx.get_op(*n).unwrap(),
                d.len()
            );
            for d in d {
                println!("    {d:?}");
            }
            dnf_nodes.entry(d.clone()).or_default().push(*n);
        }
        println!("got {} nodes", self.node_dnfs.len());
        println!("got {} DNF -> node clusters", dnf_nodes.len());
        let mut hist: BTreeMap<_, usize> = BTreeMap::new();
        for d in dnf_nodes.values() {
            *hist.entry(d.len()).or_default() += 1;
        }
        for (size, count) in hist {
            println!("{size} => {count}");
        }
        Ok(())
    }

    fn recurse(
        &mut self,
        node: Node,
        dnf: Option<DnfClause>,
    ) -> Result<(), Error> {
        let op = self.ctx.get_op(node).ok_or(Error::BadNode)?;
        if matches!(op, Op::Const(..)) {
            return Ok(());
        }

        // If we've already seen this node + DNF, then no need to recurse
        if !self.node_dnfs.entry(node).or_default().insert(dnf) {
            return Ok(());
        }
        match op {
            Op::Input(..) | Op::Var(..) => {
                // Nothing to do here
            }
            Op::Unary(_op, child) => {
                self.recurse(*child, dnf)?;
            }
            Op::Binary(BinaryOpcode::Min | BinaryOpcode::Max, lhs, rhs) => {
                let i = self.choice_id.len();
                let choice_index =
                    *self.choice_id.entry(node).or_insert(ChoiceIndex(i));

                // LHS recursion
                self.recurse(
                    *lhs,
                    Some(DnfClause {
                        root: choice_index,
                        choice: Choice::Left,
                    }),
                )?;

                // RHS recursion
                self.recurse(
                    *rhs,
                    Some(DnfClause {
                        root: choice_index,
                        choice: Choice::Right,
                    }),
                )?;
            }
            Op::Binary(_op, lhs, rhs) => {
                self.recurse(*lhs, dnf)?;
                self.recurse(*rhs, dnf)?;
            }
            Op::Const(..) => unreachable!(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_foo() {
        const PROSPERO: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/ring.txt"
        ));
        let (ctx, root) =
            crate::Context::from_text(PROSPERO.as_bytes()).unwrap();
        std::fs::write("out.dot", ctx.dot());

        let mut comp = Compiler::new(&ctx);
        comp.buildy(root);
    }
}
