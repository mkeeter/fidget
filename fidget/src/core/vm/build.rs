use std::{cell::Cell, sync::Arc};

use crate::{
    context::{BinaryOpcode, Context, Node, Op},
    eval::Family,
    vm::{alloc, op, tape, ChoiceIndex},
    Error,
};

use std::collections::{BTreeMap, BTreeSet};

/// A single choice at a particular node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ChoiceClause {
    root: Node,
    choice: usize,
}

/// Compute which nodes should be inlined
fn pick_inline(ctx: &Context, root: Node, max_weight: usize) -> BTreeSet<Node> {
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
                if w <= max_weight && node != root {
                    inline.insert(node);
                }
            }
        }
    }
    inline
}

#[derive(Clone, Default)]
struct UnorderedGroup {
    key: BTreeSet<ChoiceClause>,
    actual_nodes: BTreeSet<Node>,
    virtual_nodes: BTreeSet<Node>,
    parents: BTreeSet<ChoiceClause>,
}

#[derive(Default)]
struct OrderedGroup {
    key: BTreeSet<ChoiceClause>,
    actual_nodes: Vec<Node>,
    virtual_nodes: BTreeSet<Node>,
    parents: BTreeSet<ChoiceClause>,
}

/// Finds groups of nodes based on choices at `min` / `max` operations
fn find_groups(
    ctx: &Context,
    root: Node,
    inline: &BTreeSet<Node>,
) -> Vec<UnorderedGroup> {
    struct Action {
        node: Node,
        choice: Option<ActiveChoice>,
        child: bool,
    }
    #[derive(Clone, Debug)]
    struct ActiveChoice {
        root: Node,
        next: Arc<Cell<usize>>,
    }
    let mut todo: Vec<Action> = vec![Action {
        node: root,
        choice: None,
        child: true,
    }];
    let mut node_parents: BTreeMap<Node, BTreeSet<ChoiceClause>> =
        BTreeMap::new();
    let mut virtual_nodes: BTreeSet<Node> = BTreeSet::new();
    let mut node_choices: BTreeMap<Node, BTreeSet<Option<ChoiceClause>>> =
        BTreeMap::new();

    while let Some(Action {
        node,
        choice,
        child,
    }) = todo.pop()
    {
        // Skip inlined nodes
        if inline.contains(&node) {
            continue;
        }
        let op = ctx.get_op(node).unwrap();

        // Special handling to flatten out commutative min/max trees
        if child {
            if let Some(choice) = &choice {
                let root_op = ctx.get_op(choice.root).unwrap();
                let Op::Binary(prev_opcode, ..) = root_op else { panic!() };
                if let Op::Binary(this_opcode, lhs, rhs) = op {
                    if prev_opcode == this_opcode {
                        // Special case: skip this node in the tree, because
                        // we're inserting its children instead
                        todo.push(Action {
                            node: *rhs,
                            choice: Some(choice.clone()),
                            child: true,
                        });
                        todo.push(Action {
                            node: *lhs,
                            choice: Some(choice.clone()),
                            child: true,
                        });
                        continue;
                    }
                }
            }
        }

        // If this is a child, then we increment the choice clause counter here;
        // we always decrement it when reading.  This seems weird, but is the
        // easiest way to make the value correct for both child nodes (i.e.
        // direct arguments to the n-ary operation) and lower-down nodes where
        // `child = false`.
        if child {
            if let Some(choice) = &choice {
                choice.next.set(choice.next.get() + 1);
            }
        }

        let choice_clause = choice.as_ref().map(|c| ChoiceClause {
            root: c.root,
            choice: c.next.get() - 1,
        });
        // If we've already seen this node + choice, then no need to recurse
        if !node_choices.entry(node).or_default().insert(choice_clause) {
            continue;
        }

        // If this node is a direct child of an n-ary operation, then record it
        // in the node_parents array.
        if child {
            if let Some(choice_clause) = &choice_clause {
                node_parents.entry(node).or_default().insert(*choice_clause);
            }
        }

        match op {
            Op::Binary(BinaryOpcode::Min | BinaryOpcode::Max, lhs, rhs) => {
                // Special case: min(reg, imm) and min(imm, reg) both become
                // MinRegImm nodes, so we swap Left and Right in that case
                let (lhs, rhs) =
                    if matches!(ctx.get_op(*lhs).unwrap(), Op::Const(..)) {
                        (rhs, lhs)
                    } else {
                        (lhs, rhs)
                    };

                virtual_nodes.insert(node);

                let d = Arc::new(Cell::new(0));

                // RHS recursion; we push this first so that the LHS gets the
                // first choice (since this is a stack, not a queue).
                todo.push(Action {
                    node: *rhs,
                    choice: Some(ActiveChoice {
                        root: node,
                        next: d.clone(),
                    }),
                    child: true,
                });

                // LHS recursion
                todo.push(Action {
                    node: *lhs,
                    choice: Some(ActiveChoice {
                        root: node,
                        next: d.clone(),
                    }),
                    child: true,
                });
            }
            op => {
                for c in op.iter_children() {
                    todo.push(Action {
                        node: c,
                        choice: choice.clone(),
                        child: false,
                    });
                }
            }
        }
    }

    // At this point, we've populated node_choices and node_parents with
    // everything that we need to build groups.

    // Any root-accessible node must always be alive, so we can collapse its
    // conditional down to just `None`
    for dnf in node_choices.values_mut() {
        if dnf.contains(&None) {
            dnf.retain(Option::is_none)
        }
    }

    // Swap around, fron node -> DNF to DNF -> [Nodes]
    let mut dnf_nodes: BTreeMap<_, BTreeSet<Node>> = BTreeMap::new();
    for (n, d) in node_choices {
        if d.contains(&None) {
            assert_eq!(d.len(), 1);
            dnf_nodes.entry(BTreeSet::new())
        } else {
            assert!(!d.is_empty());
            dnf_nodes.entry(d.into_iter().map(Option::unwrap).collect())
        }
        .or_default()
        .insert(n);
    }

    let mut out = vec![];
    for (key, values) in dnf_nodes.into_iter() {
        // Each group only has one output node; we find it here and copy the
        // parents from that node to the group.
        let mut parents = BTreeSet::new();
        for node in &values {
            if let Some(p) = node_parents.get(node) {
                assert!(parents.is_empty());
                parents = p.clone();
            }
        }
        // Skip virtual nodes, which serve as the root for n-ary trees but
        // aren't explicitly in the node list.  This means that some groups
        // could be empty, because they only contain a virtual node; in that
        // case, the work for that node would be done in child groups that
        // populate it.
        let (virtual_nodes, actual_nodes) =
            values.into_iter().partition(|n| virtual_nodes.contains(n));
        out.push(UnorderedGroup {
            key,
            actual_nodes,
            virtual_nodes,
            parents,
        });
    }
    out
}

fn insert_inline_nodes(
    ctx: &Context,
    inline: &BTreeSet<Node>,
    group: &mut BTreeSet<Node>,
) {
    let mut todo = BTreeSet::new();
    for node in group.iter() {
        let op = ctx.get_op(*node).unwrap();
        todo.extend(op.iter_children().filter(|c| inline.contains(c)));
    }
    let mut todo: Vec<_> = todo.into_iter().collect();
    while let Some(node) = todo.pop() {
        if group.insert(node) {
            todo.extend(ctx.get_op(node).unwrap().iter_children());
        }
    }
}

/// Find any nodes within this group whose children are **not** in the group
fn find_globals<'a>(
    ctx: &'a Context,
    group: &'a UnorderedGroup,
) -> impl Iterator<Item = Node> + 'a {
    group
        .actual_nodes
        .iter()
        .flat_map(|node| {
            ctx.get_op(*node).unwrap().iter_children().filter(|c| {
                !(group.actual_nodes.contains(c)
                    || group.virtual_nodes.contains(c))
            })
        })
        .chain(group.virtual_nodes.iter().cloned())
}

fn compute_group_order(
    ctx: &Context,
    root: Node,
    groups: &[UnorderedGroup],
    globals: &BTreeSet<Node>,
) -> Vec<usize> {
    // Mapping for any global node to the group that contains it.
    let global_node_to_group: BTreeMap<Node, usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(i, g)| {
            g.actual_nodes
                .iter()
                .chain(g.virtual_nodes.iter())
                .filter(|n| globals.contains(n))
                .map(move |n| (*n, i))
        })
        .collect();

    // Compute the child groups of every individual group
    let mut children: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (g, group) in groups.iter().enumerate() {
        let map: BTreeSet<_> = group
            .actual_nodes
            .iter()
            .flat_map(|node| {
                ctx.get_op(*node)
                    .unwrap()
                    .iter_children()
                    .filter_map(|c| global_node_to_group.get(&c).cloned())
                    .filter(|cg| *cg != g)
            })
            .collect();
        children.insert(g, map);
    }

    // Add parent-child relationships based on virtual nodes
    for (g, group) in groups.iter().enumerate() {
        for p in &group.parents {
            children
                .entry(global_node_to_group[&p.root])
                .or_default()
                .insert(g);
        }
    }

    // For every group, count how many parents it has.  When flattening,
    // we'll add a group once all of its parents have been seen.
    let mut parent_count: BTreeMap<usize, usize> = BTreeMap::new();
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
        if *parent_count.get(&group).unwrap_or(&0) > 0 || !seen.insert(group) {
            continue;
        }
        for &child in children[&group].iter() {
            todo.push(child);
            *parent_count.get_mut(&child).unwrap() -= 1;
        }
        ordered_groups.push(group);
    }
    // At this point, ordered_groups is a list of groups from root to leafs.
    assert_eq!(ordered_groups[0], global_node_to_group[&root]);
    let mut group_order = vec![0; ordered_groups.len()];
    for (i, g) in ordered_groups.into_iter().enumerate() {
        group_order[g] = i;
    }
    group_order
}

fn apply_group_order(
    groups: Vec<UnorderedGroup>,
    group_order: Vec<usize>,
) -> Vec<UnorderedGroup> {
    let mut groups_out = vec![Default::default(); group_order.len()];
    for (i, g) in groups.into_iter().enumerate() {
        let j = group_order[i];
        groups_out[j] = g;
    }
    groups_out
}

fn sort_nodes(ctx: &Context, group: UnorderedGroup) -> OrderedGroup {
    let mut parent_count: BTreeMap<Node, usize> = BTreeMap::new();

    // Find starting roots for this cluster, which are defined as any
    // node that's not an input to other nodes in the cluster.
    let children: BTreeSet<Node> = group
        .actual_nodes
        .iter()
        .flat_map(|n| ctx.get_op(*n).unwrap().iter_children())
        .filter(|n| {
            group.actual_nodes.contains(n) || group.virtual_nodes.contains(n)
        })
        .collect();

    let start: Vec<Node> = group
        .actual_nodes
        .iter()
        .filter(|n| !children.contains(n))
        .cloned()
        .collect();

    // Compute the number of within-group parents of each group node
    let mut seen = BTreeSet::new();
    let mut todo = start.clone();
    while let Some(n) = todo.pop() {
        if !seen.insert(n) {
            continue;
        }
        for child in ctx
            .get_op(n)
            .unwrap()
            .iter_children()
            .filter(|n| group.actual_nodes.contains(n))
        {
            *parent_count.entry(child).or_default() += 1;
            todo.push(child);
        }
    }

    // Great, we can now generate an ordering within the group
    let mut ordered_nodes = vec![];
    let mut seen = BTreeSet::new();
    let mut todo = start;
    while let Some(n) = todo.pop() {
        if *parent_count.get(&n).unwrap_or(&0) > 0 || !seen.insert(n) {
            continue;
        }
        for child in ctx
            .get_op(n)
            .unwrap()
            .iter_children()
            .filter(|n| group.actual_nodes.contains(n))
        {
            todo.push(child);
            *parent_count.get_mut(&child).unwrap() -= 1;
        }
        ordered_nodes.push(n);
    }
    OrderedGroup {
        key: group.key,
        actual_nodes: ordered_nodes,
        virtual_nodes: group.virtual_nodes,
        parents: group.parents,
    }
}

/// Eliminates any `Load` operation which already has the value in the register
fn eliminate_forward_loads(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    // Records the active register -> memory mapping.  If a register is only
    // used as a local value, then this map does not contain its value.
    let mut reg_mem = BTreeMap::new();
    let mut out_groups = vec![];

    for group in group_tapes.iter().rev() {
        let mut out = vec![];
        for op in group.iter().rev() {
            match *op {
                op::Op::Load { reg, mem } => {
                    // If the reg-mem binding matches, then we don't need this
                    // Load operation in the tape.
                    if Some(&mem) == reg_mem.get(&reg) {
                        continue; // skip this node
                    } else {
                        // This register is now invalidated, since it may be
                        // bound to a different memory address; we can't be
                        // *sure* of that binding, though, because groups can be
                        // deactivated.
                        reg_mem.remove(&reg);
                    }
                }
                op::Op::Store { reg, mem } => {
                    // Update the reg-mem binding based on this Store
                    reg_mem.insert(reg, mem);
                }
                _ => {
                    if let Some(out) = op.out_reg() {
                        // All other operations invalidate the reg-mem binding
                        reg_mem.remove(&out);
                    }
                }
            }
            out.push(*op);
        }
        out.reverse();
        out_groups.push(out);
    }
    out_groups.reverse();
    out_groups
}

/// Eliminates any `Load` operation which does not use the register
fn eliminate_reverse_loads(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    // Next up, do a pass and eliminate any Load operation which isn't used
    //
    // We do this by walking in reverse-evaluation order through each group,
    // keeping track of which registers are active (i.e. have been used as
    // inputs since they were last written).
    //
    // If there is a load operation to a non-active register, we can skip it.
    let mut out_groups = vec![];
    let mut active = BTreeSet::new();
    active.insert(0); // Register 0 starts active
    for group in group_tapes.iter() {
        let mut out = vec![];
        for op in group {
            if let op::Op::Load { reg, .. } = op {
                if !active.contains(reg) {
                    continue; // skip this Load
                }
            } else {
                if let Some(out) = op.out_reg() {
                    active.remove(&out);
                }
                active.extend(op.input_reg_iter());
            }
            out.push(*op);
        }
        out_groups.push(out);
        active.clear();
    }
    out_groups
}

/// Eliminates any `Store` operation which does not have a matching `Load`
fn eliminate_reverse_stores(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    let mut out_groups = vec![];
    let mut active = BTreeSet::new();
    for group in group_tapes.iter() {
        let mut out = vec![];
        for op in group {
            if let op::Op::Load { mem, .. } = op {
                active.insert(mem);
            } else if let op::Op::Store { mem, .. } = op {
                if !active.remove(mem) {
                    continue; // skip this node
                }
            }
            out.push(*op);
        }
        out_groups.push(out);
    }
    out_groups
}

/// Eliminate any `Load` operations that occur before a `Store`
fn eliminate_dead_loads(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    let mut out_groups = vec![];
    let mut stored = BTreeSet::new();
    for group in group_tapes.iter().rev() {
        let mut out = vec![];
        for op in group.iter().rev() {
            match op {
                op::Op::Store { mem, .. }
                | op::Op::MaxRegImmChoice { mem, .. }
                | op::Op::MinRegImmChoice { mem, .. }
                | op::Op::MaxRegRegChoice { mem, .. }
                | op::Op::MinRegRegChoice { mem, .. } => {
                    stored.insert(mem);
                }
                op::Op::Load { mem, .. } => {
                    if !stored.contains(mem) {
                        continue;
                    }
                }
                _ => (),
            }
            out.push(*op);
        }
        out.reverse();
        out_groups.push(out);
    }
    out_groups.reverse();
    out_groups
}

pub fn buildy<F: Family>(
    ctx: &Context,
    root: Node,
    inline: usize,
) -> Result<tape::TapeData<F>, Error> {
    if let Some(c) = ctx.const_value(root).unwrap() {
        let t = tape::ChoiceTape {
            tape: vec![op::Op::CopyImm {
                out: 0,
                imm: c as f32,
            }],
            choices: vec![],
        };
        return Ok(tape::TapeData::new(vec![t], BTreeMap::new()));
    }

    let inline = pick_inline(ctx, root, inline);

    let mut groups = find_groups(ctx, root, &inline);

    // Remove duplicate parents, i.e. if we do `max(max(x, y), max(x, z))`, we
    // only need to store `x` once in the n-ary operation.
    for g in &mut groups {
        let mut new_parents = BTreeSet::new();
        let mut roots = BTreeSet::new();
        for v in g.parents.iter() {
            if roots.insert(v.root) {
                new_parents.insert(*v);
            }
        }
        g.parents = new_parents;
    }
    // TODO compress choices here, since we may have removed some of them

    // For each DNF, add all of the inlined nodes
    for group in groups.iter_mut() {
        insert_inline_nodes(ctx, &inline, &mut group.actual_nodes);
    }
    let groups = groups; // remove mutability

    // Compute all of the global nodes
    let globals: BTreeSet<Node> = groups
        .iter()
        .flat_map(|g| find_globals(ctx, &g))
        .chain(std::iter::once(root))
        .collect();

    // Sanity-checking our globals and inlined variables
    assert!(globals.intersection(&inline).count() == 0);

    // Reorder the groups from root-to-leaf order
    //
    // Note that the nodes within each group remain unordered
    let group_order = compute_group_order(ctx, root, &groups, &globals);
    let groups = apply_group_order(groups, group_order);

    // Build a mapping from choice nodes to indices in the choice data array
    let mut choices_per_node: BTreeMap<Node, usize> = BTreeMap::new();
    for t in &groups {
        for k in &t.key {
            let c = choices_per_node.entry(k.root).or_default();
            *c = (*c).max(k.choice);
        }
    }
    let mut node_to_choice_index: BTreeMap<Node, usize> = BTreeMap::new();
    let mut offset = 0;
    for (node, choice_count) in choices_per_node {
        node_to_choice_index.insert(node, offset);
        offset += choice_count / 8 + 1;
    }

    // Perform node ordering within each group
    let groups: Vec<OrderedGroup> =
        groups.into_iter().map(|g| sort_nodes(ctx, g)).collect();

    let mut first_seen = BTreeMap::new();
    for (i, group) in groups.iter().enumerate().rev() {
        for c in &group.parents {
            first_seen.entry(c.root).or_insert(i);
        }
    }

    let mut group_tapes = vec![];
    let mut alloc = alloc::RegisterAllocator::new(ctx, root, F::REG_LIMIT);
    for (i, group) in groups.iter().enumerate() {
        // Because the group is sorted, the output node will be the first one
        //
        // If there are no actual nodes, then there must be a single virtual
        // node and that must be the output of the group.
        let out = if group.actual_nodes.is_empty() {
            assert_eq!(group.virtual_nodes.len(), 1);
            group.virtual_nodes.iter().cloned().next().unwrap()
        } else {
            group.actual_nodes[0]
        };

        // Add virtual nodes for any n-ary operations associated with this group
        for k in group.parents.iter() {
            let c = node_to_choice_index[&k.root];
            alloc.nary_op(k.root, out, ChoiceIndex::new(c, k.choice));
            if first_seen[&k.root] == i {
                alloc.release_nary_op(k.root);
            }
        }

        for node in group.actual_nodes.iter() {
            alloc.op(*node);
        }
        for node in group.virtual_nodes.iter() {
            alloc.virtual_op(*node);
        }
        let tape = alloc.finalize();
        group_tapes.push(tape);
    }

    // Hooray, we've got group tapes!
    //
    // Tapes were planned conservatively, assuming that global values have to be
    // moved to RAM (rather than persisting in registers).  We're going to do a
    // few cleanup passes to remove dead Load and Store operations introduced by
    // that assumption.

    for pass in [
        eliminate_forward_loads,
        eliminate_reverse_loads,
        eliminate_reverse_stores,
        eliminate_dead_loads,
    ] {
        group_tapes = pass(&group_tapes);
    }

    let gt = group_tapes
        .into_iter()
        .zip(groups.into_iter())
        .map(|(g, k)| tape::ChoiceTape {
            tape: g,
            choices: k
                .key
                .into_iter()
                .map(|d| {
                    ChoiceIndex::new(node_to_choice_index[&d.root], d.choice)
                })
                .collect(),
        })
        .filter(|t| !t.tape.is_empty())
        .collect::<Vec<_>>();

    /*
    println!("------------------------------------------------------------");
    for g in gt.iter().rev() {
        println!("{:?}", g.choices);
        println!("{}", g.choices.len());
        for op in g.tape.iter().rev() {
            println!("{op:?}");
        }
        println!();
    }
    */
    Ok(tape::TapeData::new(gt, alloc.var_names()))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dead_load_store() {
        const PROSPERO: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/prospero.vm"
        ));
        let (ctx, root) =
            crate::Context::from_text(PROSPERO.as_bytes()).unwrap();

        // When planning for the interpreter, we get tons of registers, so we
        // should never see a Load or Store operation.
        let t = buildy::<crate::vm::Eval>(&ctx, root, 9).unwrap();
        assert!(t.slot_count() <= 255);
        for op in t.data.iter().flat_map(|t| t.tape.iter()) {
            assert!(!matches!(op, op::Op::Load { .. } | op::Op::Store { .. }));
        }
    }

    #[test]
    fn test_groups() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();

        let one = ctx.constant(1.0);
        let max1 = ctx.max(x, one).unwrap();
        let sum = ctx.add(max1, y).unwrap();
        let max2 = ctx.max(sum, z).unwrap();

        // max(max(x, 1) + y, z)

        let r = find_groups(&ctx, max2, &BTreeSet::new());
        let roots = r
            .iter()
            .flat_map(|g| g.key.iter().map(|k| k.root))
            .collect::<BTreeSet<Node>>();
        assert_eq!(roots.len(), 2);
    }
}
