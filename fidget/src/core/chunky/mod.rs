use crate::{
    context::{BinaryOpcode, Context, Node, Op, UnaryOpcode, VarNode},
    eval::tracing::Choice,
    Error,
};

use std::collections::{BTreeMap, BTreeSet};

/// Globally unique index for a particular choice node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ChoiceIndex(usize);

/// Globally unique index for a particular DNF group
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct GroupIndex(usize);

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

    /// Compute which nodes should be inlined
    fn pick_inline(
        ctx: &Context,
        root: Node,
        max_weight: usize,
    ) -> BTreeSet<Node> {
        enum Action {
            Down,
            Up,
        }

        // Recursively assign weights to nodes in the graph, using a loop with
        // manual recursion to avoid blowing the stack.
        //
        // A rank is `sum(rank(c) for c in children) + 1`, i.e. the total number
        // of operations required for a node.
        let mut todo = vec![(Action::Down, root)];
        let mut weights = BTreeMap::new();
        let mut inline = BTreeSet::new();
        while let Some((a, node)) = todo.pop() {
            if weights.contains_key(&node) {
                continue;
            }
            match a {
                Action::Down => {
                    todo.push((Action::Up, node));
                    let op = ctx.get_op(node).unwrap();
                    for c in op.iter_children() {
                        todo.push((Action::Down, c));
                    }
                }
                Action::Up => {
                    let op = ctx.get_op(node).unwrap();
                    let w: usize = op
                        .iter_children()
                        .map(|c| weights.get(&c).unwrap())
                        .sum::<usize>()
                        + 1;
                    weights.insert(node, w);
                    if w <= max_weight {
                        inline.insert(node);
                    }
                }
            }
        }
        inline
    }

    pub fn buildy(&mut self, root: Node) -> Result<(), Error> {
        let inline = Self::pick_inline(self.ctx, root, 7);
        println!("got {} inline nodes", inline.len());

        let mut todo = vec![(root, None)];
        while let Some((node, dnf)) = todo.pop() {
            if inline.contains(&node) {
                continue;
            }
            let op = self.ctx.get_op(node).ok_or(Error::BadNode)?;
            if matches!(op, Op::Const(..)) {
                continue;
            }

            // If we've already seen this node + DNF, then no need to recurse
            if !self.node_dnfs.entry(node).or_default().insert(dnf) {
                continue;
            }
            match op {
                Op::Input(..) | Op::Var(..) => {
                    // Nothing to do here
                }
                Op::Unary(_op, child) => {
                    todo.push((*child, dnf));
                }
                Op::Binary(BinaryOpcode::Min | BinaryOpcode::Max, lhs, rhs) => {
                    let i = self.choice_id.len();
                    let choice_index =
                        *self.choice_id.entry(node).or_insert(ChoiceIndex(i));

                    // LHS recursion
                    todo.push((
                        *lhs,
                        Some(DnfClause {
                            root: choice_index,
                            choice: Choice::Left,
                        }),
                    ));

                    // RHS recursion
                    todo.push((
                        *rhs,
                        Some(DnfClause {
                            root: choice_index,
                            choice: Choice::Right,
                        }),
                    ));
                }
                Op::Binary(_op, lhs, rhs) => {
                    todo.push((*lhs, dnf));
                    todo.push((*rhs, dnf));
                }
                Op::Const(..) => unreachable!(),
            }
        }

        // Swap around, fron node -> DNF to DNF -> [Nodes]
        let mut dnf_nodes: BTreeMap<_, BTreeSet<Node>> = BTreeMap::new();
        for (n, d) in &self.node_dnfs {
            dnf_nodes.entry(d.clone()).or_default().insert(*n);
        }

        // For each DNF, add all of the inlined nodes
        for nodes in dnf_nodes.values_mut() {
            let mut todo = BTreeSet::new();
            for node in nodes.iter() {
                let op = self.ctx.get_op(*node).unwrap();
                todo.extend(op.iter_children().filter(|c| inline.contains(c)));
            }
            let mut todo: Vec<_> = todo.into_iter().collect();
            while let Some(node) = todo.pop() {
                if nodes.insert(node) {
                    todo.extend(self.ctx.get_op(node).unwrap().iter_children());
                }
            }
        }

        // Any node which reaches out of a cluster must be from another cluster.
        let mut globals = BTreeSet::new();
        for nodes in dnf_nodes.values() {
            globals.extend(nodes.iter().flat_map(|node| {
                self.ctx
                    .get_op(*node)
                    .unwrap()
                    .iter_children()
                    .filter(|c| !nodes.contains(c))
            }));
        }
        assert!(globals.intersection(&inline).count() == 0);
        assert!(!globals.contains(&root));
        globals.insert(root);

        // Mapping for any global node to the group that contains it.
        let global_node_to_group: BTreeMap<Node, GroupIndex> = dnf_nodes
            .values()
            .enumerate()
            .flat_map(|(i, nodes)| {
                nodes
                    .iter()
                    .filter(|n| globals.contains(n))
                    .map(move |n| (*n, GroupIndex(i)))
            })
            .collect();

        // Compute the child groups of every individual group
        let mut children: BTreeMap<GroupIndex, BTreeSet<GroupIndex>> =
            BTreeMap::new();
        for (g, nodes) in dnf_nodes.values().enumerate() {
            let g = GroupIndex(g);
            let map: BTreeSet<GroupIndex> = nodes
                .iter()
                .flat_map(|node| {
                    self.ctx
                        .get_op(*node)
                        .unwrap()
                        .iter_children()
                        .filter_map(|c| global_node_to_group.get(&c).cloned())
                        .filter(|cg| *cg != g)
                })
                .collect();
            children.insert(g, map);
        }

        // For every group, count how many parents it has.  When flattening,
        // we'll add a group once all of its parents have been seen.
        let mut parent_count: BTreeMap<GroupIndex, usize> = BTreeMap::new();
        let mut todo = vec![global_node_to_group[&root]];
        let mut seen = BTreeSet::new();
        while let Some(group) = todo.pop() {
            if !seen.insert(group) {
                continue;
            }
            for &child in children[&group].iter() {
                *parent_count.entry(child).or_default() += 1;
                todo.push(child);
            }
        }

        // Now that we've populated our parents, flatten the graph
        let mut ordered_groups = vec![];
        let mut todo = vec![global_node_to_group[&root]];
        let mut seen = BTreeSet::new();
        while let Some(group) = todo.pop() {
            if *parent_count.get(&group).unwrap_or(&0) > 0
                || !seen.insert(group)
            {
                continue;
            }
            for &child in children[&group].iter() {
                todo.push(child);
                *parent_count.get_mut(&child).unwrap() -= 1;
            }
            ordered_groups.push(group);
        }

        for o in &ordered_groups {
            println!("=======================");
            let g = dnf_nodes.values().nth(o.0).unwrap();
            for n in g {
                if globals.contains(n) {
                    print!("GLOBAL ");
                }
                if inline.contains(n) {
                    print!("INLINE ");
                }
                println!("{:?}", self.ctx.get_op(*n).unwrap());
            }
        }

        // At this point, ordered_groups is a list of groups from root to leafs.
        assert_eq!(ordered_groups[0], global_node_to_group[&root]);

        ////////////////////////////////////////////////////////////////////////

        println!("got {} nodes", self.node_dnfs.len());
        println!("got {} DNF -> node clusters", dnf_nodes.len());
        println!("got {} globals", globals.len());

        let mut hist: BTreeMap<_, usize> = BTreeMap::new();
        for d in dnf_nodes.values() {
            *hist.entry(d.len()).or_default() += 1;
        }
        println!(" OPS | COUNT");
        println!("-----|------");
        for (size, count) in &hist {
            println!(" {size:>3} | {count:<5}");
        }
        println!("total functions: {}", hist.values().sum::<usize>());
        println!(
            "total instructions: {}",
            hist.iter().map(|(a, b)| a * b).sum::<usize>()
        );
        println!();

        let mut hist: BTreeMap<_, usize> = BTreeMap::new();
        for d in dnf_nodes.values() {
            let outputs = d.iter().filter(|n| globals.contains(n)).count();
            *hist.entry((d.len(), outputs)).or_default() += 1;
        }
        println!(" OPS | OUT | COUNT");
        println!("-----|-----|------");
        for ((size, out), count) in &hist {
            println!(" {size:>3} | {out:^3} | {count:<5}");
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
            "/../models/prospero.vm"
        ));
        let (ctx, root) =
            crate::Context::from_text(PROSPERO.as_bytes()).unwrap();
        std::fs::write("out.dot", ctx.dot());

        let mut comp = Compiler::new(&ctx);
        comp.buildy(root);
    }
}
