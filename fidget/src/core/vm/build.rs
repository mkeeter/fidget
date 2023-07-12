use std::{cell::Cell, sync::Arc};

use crate::{
    context::{BinaryOpcode, Context, Node, Op},
    eval::Family,
    vm::{self, alloc, tape, ChoiceIndex, ChoiceMask},
    Error,
};

use std::collections::{BTreeMap, BTreeSet};

/// A single choice at a particular node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ChoiceClause {
    root: Node,
    choice: usize,
}

#[derive(Clone, Default)]
struct UnorderedGroup {
    key: BTreeSet<ChoiceClause>,
    actual_nodes: BTreeSet<Node>,

    /// Nodes which belong to this group, but are in practice written by
    /// in-place operations at the end of other groups.
    ///
    /// For example, `min(x, y)` would have three groups:
    /// ```
    /// [x, InPlaceMin(out, x)]
    /// [y, InPlaceMin(out, y)]
    /// []
    /// ```
    /// The last group has a virtual node representing the result of the `min`.
    virtual_nodes: BTreeSet<Node>,

    /// Parent `min` and `max` nodes
    ///
    /// These must be written by the output of the group, and should be a subset
    /// of [`self.key`]
    parents: BTreeSet<ChoiceClause>,
}

#[derive(Default)]
struct OrderedGroup {
    key: BTreeSet<ChoiceClause>,
    actual_nodes: Vec<Node>,
    virtual_nodes: BTreeSet<Node>,

    /// Virtual input node, for when we peel off an in-place operation
    virtual_input: Option<Node>,

    parents: BTreeSet<ChoiceClause>,
}

impl OrderedGroup {
    fn output(&self) -> Node {
        // Because the group is sorted, the output node will be the first one
        //
        // If there are no actual nodes, then there must be a single virtual
        // node (either in `virtual_nodes` or `virtual_input`, but not both),
        // and that must be the output of the group.
        if self.actual_nodes.is_empty() {
            if self.virtual_nodes.len() > 0 {
                assert_eq!(self.virtual_nodes.len(), 1);
                self.virtual_nodes.iter().cloned().next().unwrap()
            } else if let Some(out) = self.virtual_input {
                out
            } else {
                panic!("no output");
            }
        } else {
            assert!(self.virtual_input.is_none());
            self.actual_nodes[0]
        }
    }
}

/// Finds groups of nodes based on choices at `min` / `max` operations
fn find_groups(ctx: &Context, root: Node) -> Vec<UnorderedGroup> {
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
        virtual_input: None,
        parents: group.parents,
    }
}

/// Eliminates any `Load` operation which already has the value in the register
fn eliminate_forward_loads(group_tapes: &[Vec<vm::Op>]) -> Vec<Vec<vm::Op>> {
    // Records the active register -> memory mapping.  If a register is only
    // used as a local value, then this map does not contain its value.
    let mut reg_mem = BTreeMap::new();
    let mut out_groups = vec![];

    for group in group_tapes.iter().rev() {
        let mut out = vec![];
        for op in group.iter().rev() {
            match *op {
                vm::Op::Load { reg, mem } => {
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
                vm::Op::Store { reg, mem } => {
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
fn eliminate_reverse_loads(group_tapes: &[Vec<vm::Op>]) -> Vec<Vec<vm::Op>> {
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
            if let vm::Op::Load { reg, .. } = op {
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
fn eliminate_reverse_stores(group_tapes: &[Vec<vm::Op>]) -> Vec<Vec<vm::Op>> {
    let mut out_groups = vec![];
    let mut active = BTreeSet::new();
    for group in group_tapes.iter() {
        let mut out = vec![];
        for op in group {
            if let vm::Op::Load { mem, .. } = op {
                active.insert(mem);
            } else if let vm::Op::Store { mem, .. } = op {
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
fn eliminate_dead_loads(group_tapes: &[Vec<vm::Op>]) -> Vec<Vec<vm::Op>> {
    let mut out_groups = vec![];
    let mut stored = BTreeSet::new();
    for group in group_tapes.iter().rev() {
        let mut out = vec![];
        for op in group.iter().rev() {
            match op {
                vm::Op::Store { mem, .. }
                | vm::Op::MaxMemImmChoice { mem, .. }
                | vm::Op::MinMemImmChoice { mem, .. }
                | vm::Op::MaxMemRegChoice { mem, .. }
                | vm::Op::MinMemRegChoice { mem, .. } => {
                    stored.insert(mem);
                }
                vm::Op::Load { mem, .. } => {
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

/// Moves the persistent value in `*Choice` operations from memory to registers
fn lower_mem_to_reg(
    group_tapes: &[Vec<vm::Op>],
    reg_limit: u8,
) -> Vec<Vec<vm::Op>> {
    use super::active::{ActiveRegisterRange, RegisterSet};

    let flat: Vec<vm::Op> =
        group_tapes.iter().flat_map(|g| g.iter()).cloned().collect();

    let mut active_register_range = ActiveRegisterRange::new();
    let mut active_registers = RegisterSet::new();
    for node in &flat {
        if let Some(r) = node.out_reg() {
            active_registers.remove(r);
        }
        for r in node.input_reg_iter() {
            active_registers.insert(r);
        }
        active_register_range.push(active_registers);
    }

    // Iterate over the list in evaluation order, tracking the lifetimes of
    // choice operations `mem` values.
    let mut first_seen: BTreeMap<u32, usize> = BTreeMap::new();
    let mut last_seen: BTreeMap<u32, usize> = BTreeMap::new();
    let mut lifetimes = vec![];
    for (i, node) in flat.iter().enumerate().rev() {
        match node {
            vm::Op::MaxMemRegChoice { mem, .. }
            | vm::Op::MinMemRegChoice { mem, .. }
            | vm::Op::MaxMemImmChoice { mem, .. }
            | vm::Op::MinMemImmChoice { mem, .. } => {
                // If we have seen a `Load` for this memory location, then a
                // store begins a new lifetime; flush out the previous
                // data.
                if let Some(last) = last_seen.remove(mem) {
                    let first = first_seen.remove(mem).unwrap();
                    lifetimes.push((*mem, last..=first));
                }

                // If this is the first time we're seeing this memory
                // location, then record the index in our map
                first_seen.entry(*mem).or_insert(i);
            }
            vm::Op::Store { mem, .. } => {
                // Same as above; a Store ends the lifetime for this memory
                // location.  However, we don't begin a new lifetime,
                // because we're only tracking Choice operations here
                if let Some(last) = last_seen.remove(mem) {
                    let first = first_seen.remove(mem).unwrap();
                    lifetimes.push((*mem, last..=first));
                }
            }
            vm::Op::Load { mem, .. } => {
                if first_seen.contains_key(mem) {
                    last_seen.insert(*mem, i);
                }
            }
            _ => (),
        }
    }
    // Flush remaining elements
    for (mem, last) in last_seen.into_iter() {
        let first = first_seen.remove(&mem).unwrap();
        lifetimes.push((mem, last..=first));
    }

    // Decide and apply register remapping
    let mut flat = flat;
    for (target_mem, range) in lifetimes {
        if let Some(new_reg) = active_register_range
            .range_query(range.clone())
            .available()
            .filter(|reg| (*reg as u8) < reg_limit)
        {
            for op in &mut flat[range.clone()] {
                *op = match *op {
                    vm::Op::MaxMemRegChoice { mem, arg, choice }
                        if mem == target_mem =>
                    {
                        vm::Op::MaxRegRegChoice {
                            reg: new_reg,
                            arg,
                            choice,
                        }
                    }
                    vm::Op::MinMemRegChoice { mem, arg, choice }
                        if mem == target_mem =>
                    {
                        vm::Op::MinRegRegChoice {
                            reg: new_reg,
                            arg,
                            choice,
                        }
                    }
                    vm::Op::MaxMemImmChoice { mem, imm, choice }
                        if mem == target_mem =>
                    {
                        vm::Op::MaxRegImmChoice {
                            reg: new_reg,
                            imm,
                            choice,
                        }
                    }
                    vm::Op::MinMemImmChoice { mem, imm, choice }
                        if mem == target_mem =>
                    {
                        vm::Op::MinRegImmChoice {
                            reg: new_reg,
                            imm,
                            choice,
                        }
                    }
                    vm::Op::Load { mem, reg } if mem == target_mem => {
                        vm::Op::CopyReg {
                            out: reg,
                            arg: new_reg,
                        }
                    }
                    op => op,
                }
            }
            active_register_range.update_range(range, new_reg.into());
        }
    }

    // Unflatten back into a nested structure
    let mut out = vec![];
    let mut slice = flat.as_slice();
    for n in group_tapes.iter().map(|g| g.len()) {
        let (first, rest) = slice.split_at(n);
        out.push(first.iter().cloned().collect());
        slice = rest;
    }
    out
}

/// Ensure that memory slots are densely packed
fn compact_memory_slots(
    group_tapes: &[Vec<vm::Op>],
    reg_limit: u8,
) -> Vec<Vec<vm::Op>> {
    let mut out = group_tapes.to_vec();
    let mut compact_mem = BTreeMap::new();
    for g in out.iter_mut().rev() {
        for op in g.iter_mut().rev() {
            match op {
                vm::Op::Load { mem, .. }
                | vm::Op::Store { mem, .. }
                | vm::Op::MinMemImmChoice { mem, .. }
                | vm::Op::MaxMemImmChoice { mem, .. }
                | vm::Op::MinMemRegChoice { mem, .. }
                | vm::Op::MaxMemRegChoice { mem, .. } => {
                    let next = compact_mem.len();
                    let e = compact_mem.entry(*mem).or_insert(next);
                    *mem = *e as u32 + reg_limit as u32;
                }
                _ => (),
            }
        }
    }
    out
}

pub fn buildy<F: Family>(
    ctx: &Context,
    root: Node,
) -> Result<tape::TapeData<F>, Error> {
    if let Some(c) = ctx.const_value(root).unwrap() {
        let t = tape::ChoiceTape {
            tape: vec![vm::Op::CopyImm {
                out: 0,
                imm: c as f32,
            }],
            choices: vec![],
            clear: vec![],
        };
        return Ok(tape::TapeData::new(vec![t], BTreeMap::new()));
    }

    let mut groups = find_groups(ctx, root);

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
    // Compress choices to only include active values, because some may have
    // been removed in the duplicate parent pruning step above.
    let mut compressed_choices: BTreeMap<Node, BTreeMap<usize, usize>> =
        BTreeMap::new();
    for g in &groups {
        for v in g.parents.iter() {
            let e = compressed_choices.entry(v.root).or_default();
            let next = e.len();
            e.entry(v.choice).or_insert(next);
        }
    }
    let compress = |k: ChoiceClause| {
        let choice = *compressed_choices
            .get(&k.root)
            .unwrap()
            .get(&k.choice)
            .unwrap();
        ChoiceClause {
            root: k.root,
            choice,
        }
    };
    for g in &mut groups {
        g.parents = std::mem::take(&mut g.parents)
            .into_iter()
            .map(compress)
            .collect();
        g.key = std::mem::take(&mut g.key)
            .into_iter()
            .map(compress)
            .collect();
    }

    // TODO: in some cases, nodes only end up with one choice associated with
    // them (which is weird!); we could replace this with a single Copy?

    // Compute all of the global nodes
    let globals: BTreeSet<Node> = groups
        .iter()
        .flat_map(|g| find_globals(ctx, &g))
        .chain(std::iter::once(root))
        .collect();

    // Reorder the groups from root-to-leaf order
    //
    // Note that the nodes within each group remain unordered
    let group_order = compute_group_order(ctx, root, &groups, &globals);
    let groups = apply_group_order(groups, group_order);

    // TODO: inlining here?

    // Build a mapping from choice nodes to indices in the choice data array
    let mut choices_per_node: BTreeMap<Node, usize> = BTreeMap::new();
    for t in &groups {
        for k in &t.parents {
            let c = choices_per_node.entry(k.root).or_default();
            *c = (*c).max(k.choice);
        }
    }
    let mut node_to_choice_index: BTreeMap<Node, usize> = BTreeMap::new();
    let mut offset = 0;
    for (&node, &choice_count) in &choices_per_node {
        node_to_choice_index.insert(node, offset);
        offset += choice_count / 8 + 1;
    }

    // Perform node ordering within each group
    let groups: Vec<OrderedGroup> =
        groups.into_iter().map(|g| sort_nodes(ctx, g)).collect();

    // At this point, we have an ordered set of ordered groups.  We'll do one
    // more pass here to peel off min/max nodes if their choice is a subset of
    // the group choice, for more efficient tape simplification.
    //
    // In other words, if we had a group with a multi-element key:
    // ```
    //  [ChoiceIndex { index: 0, bit: 1 }, ChoiceIndex { index: 1, bit: 1 }]
    //    MaxRegReg { reg: 3, arg: 1, choice: ChoiceIndex { index: 0, bit: 1 } }
    //    Input { out: 1, input: 1 }
    // ```
    //
    // It will split into two groups; notice that the upper group can then be
    // skipped during tape pruning.
    // ```
    // [ChoiceIndex { index: 0, bit: 1 }]
    //    MaxRegReg { reg: 3, arg: 1, choice: ChoiceIndex { index: 0, bit: 1 } }
    //
    // [ChoiceIndex { index: 0, bit: 1 }, ChoiceIndex { index: 1, bit: 1 }]
    //    Input { out: 1, input: 1 }
    // ```
    let mut new_groups = vec![];
    for mut g in groups.into_iter() {
        let mut to_remove = BTreeSet::new();
        for p in &g.parents {
            if g.key.len() > 1 {
                let key: BTreeSet<_> = [*p].into_iter().collect();
                new_groups.push(OrderedGroup {
                    key: key.clone(),
                    actual_nodes: vec![],
                    virtual_input: Some(g.output()),
                    virtual_nodes: BTreeSet::new(),
                    parents: key.clone(),
                });
                to_remove.insert(*p);
            } else {
                assert_eq!(g.key.iter().next().unwrap(), p);
            }
        }
        for r in to_remove {
            g.parents.remove(&r);
        }
        new_groups.push(g);
    }
    let groups = new_groups;

    // Prepare for register allocation!
    let mut first_seen = BTreeMap::new();
    for (i, group) in groups.iter().enumerate().rev() {
        for c in &group.parents {
            first_seen.entry(c.root).or_insert(i);
        }
    }

    // Perform register allocation, collecting group tapes
    let mut group_tapes = vec![];
    let mut alloc = alloc::RegisterAllocator::new(ctx, root, F::REG_LIMIT);
    for (i, group) in groups.iter().enumerate() {
        let out = group.output();

        // Add in-place nodes for n-ary operations associated with this group
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
    alloc.assert_done();

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

    // More optimizations!
    group_tapes = lower_mem_to_reg(&group_tapes, F::REG_LIMIT);

    // I'm not sure why spurious Store operations are left in the tape at this
    // point, but we'll do one more pass to clean them out.
    group_tapes = eliminate_reverse_stores(&group_tapes);

    // Now that we've removed a bunch of Load / Store operations, there may be
    // gaps in the memory slot map; remove them for efficiency.
    group_tapes = compact_memory_slots(&group_tapes, F::REG_LIMIT);

    // TODO: eliminate CopyReg operations?

    // Convert into ChoiceTape data, which requires remapping choice keys from
    // nodes to choice indices.
    let gt = group_tapes
        .into_iter()
        .zip(groups.into_iter())
        .map(|(g, k)| {
            let mut choices = BTreeMap::new();
            for d in k.key.into_iter() {
                let mut index = node_to_choice_index[&d.root];
                index += d.choice / 8;
                let bit = d.choice % 8;
                *choices.entry(index).or_default() |= 1 << bit;
            }
            let choices = choices
                .into_iter()
                .map(|(index, mask)| ChoiceMask {
                    index: index.try_into().unwrap(),
                    mask,
                })
                .collect();
            let clear = k
                .virtual_nodes
                .iter()
                .map(|n| {
                    let start = node_to_choice_index[n];
                    let count = choices_per_node[n];
                    start..(start + (count + 7) / 8)
                })
                .collect();
            // TODO: collect contiguous ranges here
            tape::ChoiceTape {
                tape: g,
                choices,
                clear,
            }
        })
        .filter(|t| !t.tape.is_empty())
        .collect::<Vec<_>>();

    /*
    println!("------------------------------------------------------------");
    for g in gt.iter().rev() {
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
    use crate::{vm::Eval, Context};

    const PROSPERO: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/prospero.vm"
    ));

    #[test]
    fn test_dead_load_store() {
        let (ctx, root) = Context::from_text(PROSPERO.as_bytes()).unwrap();

        // When planning for the interpreter, we get tons of registers, so we
        // should never see a Load or Store operation.
        let t = buildy::<Eval>(&ctx, root).unwrap();
        assert!(t.slot_count() <= 255, "too many slots: {}", t.slot_count());
        for op in t.data.iter() {
            assert!(!matches!(op, vm::Op::Load { .. } | vm::Op::Store { .. }));
        }
    }

    /*
    #[test]
    fn test_inlining() {
        // Manually reduced test case
        let mut ctx = Context::new();
        let y = ctx.y();
        let f1 = ctx.constant(1.0);
        let a = ctx.sub(y, f1).unwrap();
        let x = ctx.x();
        let f2 = ctx.constant(2.0);
        let b = ctx.sub(x, f1).unwrap();
        let c = ctx.neg(x).unwrap();
        let d = ctx.max(a, b).unwrap();
        let e = ctx.max(d, c).unwrap();
        let f = ctx.sub(y, f2).unwrap();
        let root = ctx.max(e, f).unwrap();

        let t0: Tape<_> = buildy::<Eval>(&ctx, root, 0).unwrap().into();
        println!("----");
        let t9: Tape<_> = buildy::<Eval>(&ctx, root, 9).unwrap().into();

        let eval0 = t0.new_point_evaluator();
        let eval9 = t9.new_point_evaluator();

        assert_eq!(
            ctx.eval_xyz(root, 0.0, 0.0, 0.0).unwrap() as f32,
            eval0.eval(0.0, 0.0, 0.0, &[]).unwrap().0
        );
        assert_eq!(
            eval0.eval(0.0, 0.0, 0.0, &[]).unwrap().0,
            eval9.eval(0.0, 0.0, 0.0, &[]).unwrap().0
        );
    }
    */

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

        let r = find_groups(&ctx, max2);
        let roots = r
            .iter()
            .flat_map(|g| g.key.iter().map(|k| k.root))
            .collect::<BTreeSet<Node>>();
        assert_eq!(roots.len(), 2);
    }
}
