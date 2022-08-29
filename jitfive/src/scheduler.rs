use std::collections::{BTreeMap, BTreeSet};

use crate::{
    compiler::{ChoiceIndex, NodeIndex, Op, VarIndex},
    context::{Context, Node},
    indexed::IndexMap,
    queue::PriorityQueue,
    scheduled::Scheduled,
};

#[derive(Copy, Clone, Ord, PartialEq, Eq, PartialOrd)]
struct Priority {
    /// Represents the mininum number of remaining nodes that use a child of
    /// this node.  If this is 1, then the target node is the last node which
    /// uses the child.
    min_child_score: usize,

    /// Same as `min_child`, but represents the _sum_ of remaining nodes that
    /// use a child of this node.  This allows us to prioritize killing two
    /// registers when possible.
    sum_child_score: usize,
}

struct Scheduler<'a> {
    ctx: &'a Context,

    /// Mappings from the context into the scheduler
    nodes: IndexMap<Node, NodeIndex>,
    vars: IndexMap<String, VarIndex>,

    /// Stores parents of a given node.  Parents are erased from this set as
    /// they are descheduled, so the length of the set serves as a "score".
    ///
    /// Children are stored implicitly in the context, i.e.
    /// ```
    /// ctx.get_op(nodes.get_by_index(n)).iter_children()
    /// ```
    parents: BTreeMap<NodeIndex, BTreeSet<NodeIndex>>,

    /// Stores whether the given node has been scheduled yet
    scheduled: BTreeSet<NodeIndex>,

    /// The output tape, which is topologically sorted
    out: Vec<(NodeIndex, Op)>,
}

impl<'a> Scheduler<'a> {
    fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            nodes: IndexMap::default(),
            vars: IndexMap::default(),
            parents: BTreeMap::new(),
            scheduled: BTreeSet::new(),
            out: vec![],
        }
    }
    fn run_simple(&mut self, root: Node) {
        let mut todo = vec![root];
        let mut seen = BTreeSet::new();

        // Build the graph, converting from `Node` to `NodeIndex`
        while let Some(node) = todo.pop() {
            if !seen.insert(node) {
                continue;
            }
            let index = self.nodes.insert(node);
            let op = self.ctx.get_op(node).unwrap();
            for child in op.iter_children() {
                let child_index = self.nodes.insert(child);
                todo.push(child);
            }
            use crate::context::Op as CtxOp;
            let out = match op {
                CtxOp::Unary(op, lhs) => {
                    Op::Unary(*op, self.nodes.get_by_value(*lhs).unwrap())
                }
                CtxOp::Binary(op, lhs, rhs) => Op::Binary(
                    *op,
                    self.nodes.get_by_value(*lhs).unwrap(),
                    self.nodes.get_by_value(*rhs).unwrap(),
                ),
                CtxOp::BinaryChoice(op, lhs, rhs, _c) => Op::BinaryChoice(
                    *op,
                    self.nodes.get_by_value(*lhs).unwrap(),
                    self.nodes.get_by_value(*rhs).unwrap(),
                    ChoiceIndex::from(0),
                ),
                CtxOp::Const(i) => Op::Const(i.0),
                CtxOp::Var(v) => Op::Var(self.vars.insert(
                    self.ctx.get_var_by_index(*v).unwrap().to_string(),
                )),
            };
            self.out.push((index, out));
        }
        self.out.reverse();
    }

    fn run(&mut self, root: Node) {
        let mut todo = vec![root];
        let mut seen = BTreeSet::new();
        let mut queue = PriorityQueue::new();

        // Build the graph, converting from `Node` to `NodeIndex`
        while let Some(node) = todo.pop() {
            if !seen.insert(node) {
                continue;
            }
            let index = self.nodes.insert(node);
            let op = self.ctx.get_op(node).unwrap();
            for child in op.iter_children() {
                let child_index = self.nodes.insert(child);
                self.parents.entry(child_index).or_default().insert(index);
                todo.push(child);
            }
            // If this node has no children, it's available to be scheduled
            // right away. It will kill no registers, so it should be the lowest
            // possible priority.
            if op.iter_children().next().is_none() {
                queue.insert_or_update(
                    index,
                    Priority {
                        min_child_score: usize::MAX,
                        sum_child_score: usize::MAX,
                    },
                );
            }
        }

        while let Some(index) = queue.pop() {
            // This node is now scheduled!
            self.scheduled.insert(index);

            let node = *self.nodes.get_by_index(index).unwrap();
            let op = self.ctx.get_op(node).unwrap();

            use crate::context::Op as CtxOp;
            let out = match op {
                CtxOp::Unary(op, lhs) => {
                    Op::Unary(*op, self.nodes.get_by_value(*lhs).unwrap())
                }
                CtxOp::Binary(op, lhs, rhs) => Op::Binary(
                    *op,
                    self.nodes.get_by_value(*lhs).unwrap(),
                    self.nodes.get_by_value(*rhs).unwrap(),
                ),
                CtxOp::BinaryChoice(op, lhs, rhs, _c) => Op::BinaryChoice(
                    *op,
                    self.nodes.get_by_value(*lhs).unwrap(),
                    self.nodes.get_by_value(*rhs).unwrap(),
                    ChoiceIndex::from(0),
                ),
                CtxOp::Const(i) => Op::Const(i.0),
                CtxOp::Var(v) => Op::Var(self.vars.insert(
                    self.ctx.get_var_by_index(*v).unwrap().to_string(),
                )),
            };
            self.out.push((index, out));

            for child in op.iter_children() {
                let child_index = self.nodes.get_by_value(child).unwrap();

                // Remove the newly-scheduled node from the parents set of its
                // child node.
                let parents = self.parents.get_mut(&child_index).unwrap();
                let r = parents.remove(&index);
                assert!(r);

                // The child register is now one parent away from being
                // killable, so we update the priority on all its parents
                let parents = self.parents.get(&child_index).unwrap();
                for &p in parents.iter() {
                    if let Some(priority) = self.get_priority(p) {
                        queue.insert_or_update(p, priority);
                    }
                }
            }

            // Then, do the same for parents of this node, some of which may be
            // schedulable now!
            if let Some(parents) = self.parents.get(&index) {
                for &p in parents.iter() {
                    if let Some(priority) = self.get_priority(p) {
                        queue.insert_or_update(p, priority);
                    }
                }
            }
        }
    }

    fn get_priority(&self, index: NodeIndex) -> Option<Priority> {
        // A node is schedulable if all of its children have been scheduled
        // Otherwise, we exit early.
        let node = *self.nodes.get_by_index(index).unwrap();
        let op = self.ctx.get_op(node).unwrap();

        let mut min_child_score = usize::MAX;
        let mut sum_child_score = 0;

        for child in op.iter_children() {
            let child_index = self.nodes.get_by_value(child).unwrap();
            // Early exit if this child hasn't been scheduled
            if !self.scheduled.contains(&child_index) {
                return None;
            }
            // This node will be scored based on how closely its children are to
            // being done, i.e. how many of _their_ parents are still active.
            let parents = self.parents.get(&child_index).unwrap();
            min_child_score = min_child_score.min(parents.len());
            sum_child_score += parents.len();
        }
        Some(Priority {
            min_child_score,
            sum_child_score,
        })
    }
}

pub fn schedule(ctx: &Context, root: Node) -> Scheduled {
    let mut scheduler = Scheduler::new(ctx);
    scheduler.run_simple(root);
    Scheduled::new(
        scheduler.out,
        scheduler.vars,
        scheduler.nodes.get_by_value(root).unwrap(),
    )
}
