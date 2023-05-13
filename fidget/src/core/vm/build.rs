use crate::{
    context::{BinaryOpcode, Context, Node, Op},
    eval::{Choice, Family},
    vm::{alloc, op, tape},
    Error,
};

use std::collections::{BTreeMap, BTreeSet};

/// A single choice at a particular node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct DnfClause {
    root: Node,
    choice: Choice,
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
                if w <= max_weight {
                    inline.insert(node);
                }
            }
        }
    }
    inline
}

/// Finds groups of nodes based on choices at `min` / `max` operations
///
/// Returns a tuple of `(keys, values)`.  Each item in `keys` is a set of
/// choices that activate the corresponding set of nodes in `values`; as a
/// special case, nodes which are always accessible are indicated with an empty
/// set.
fn find_groups(
    ctx: &Context,
    root: Node,
    inline: &BTreeSet<Node>,
) -> (Vec<BTreeSet<DnfClause>>, Vec<BTreeSet<Node>>) {
    // Conditional that activates a particular node
    //
    // This is recorded as an `OR` of multiple [`DnfClause`] values
    let mut todo = vec![(root, None)];
    let mut node_dnfs: BTreeMap<Node, BTreeSet<Option<DnfClause>>> =
        BTreeMap::new();

    while let Some((node, dnf)) = todo.pop() {
        if inline.contains(&node) {
            continue;
        }
        let op = ctx.get_op(node).unwrap();

        // If we've already seen this node + DNF, then no need to recurse
        if !node_dnfs.entry(node).or_default().insert(dnf) {
            continue;
        }
        match op {
            Op::Input(..) | Op::Var(..) | Op::Const(..) => {
                // Nothing to do here
            }
            Op::Unary(_op, child) => {
                todo.push((*child, dnf));
            }
            Op::Binary(BinaryOpcode::Min | BinaryOpcode::Max, lhs, rhs) => {
                // LHS recursion
                todo.push((
                    *lhs,
                    Some(DnfClause {
                        root: node,
                        choice: Choice::Left,
                    }),
                ));

                // RHS recursion
                todo.push((
                    *rhs,
                    Some(DnfClause {
                        root: node,
                        choice: Choice::Right,
                    }),
                ));
            }
            Op::Binary(_op, lhs, rhs) => {
                todo.push((*lhs, dnf));
                todo.push((*rhs, dnf));
            }
        }
    }

    // Any root-accessible node must always be alive, so we can collapse its
    // conditional down to just `None`
    for dnf in node_dnfs.values_mut() {
        if dnf.len() > 1 && dnf.contains(&None) {
            dnf.retain(Option::is_none)
        }
    }

    // Swap around, fron node -> DNF to DNF -> [Nodes]
    let mut dnf_nodes: BTreeMap<_, BTreeSet<Node>> = BTreeMap::new();
    for (n, d) in node_dnfs {
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

    let mut keys = vec![];
    let mut groups = vec![];
    for (k, g) in dnf_nodes.into_iter() {
        keys.push(k);
        groups.push(g);
    }
    (keys, groups)
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
    group: &'a BTreeSet<Node>,
) -> impl Iterator<Item = Node> + 'a {
    group.iter().flat_map(|node| {
        ctx.get_op(*node)
            .unwrap()
            .iter_children()
            .filter(|c| !group.contains(c))
    })
}

fn compute_group_order(
    ctx: &Context,
    root: Node,
    groups: &[BTreeSet<Node>],
    globals: &BTreeSet<Node>,
) -> Vec<usize> {
    // Mapping for any global node to the group that contains it.
    let global_node_to_group: BTreeMap<Node, usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(i, nodes)| {
            nodes
                .iter()
                .filter(|n| globals.contains(n))
                .map(move |n| (*n, i))
        })
        .collect();

    // Compute the child groups of every individual group
    let mut children: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (g, nodes) in groups.iter().enumerate() {
        let map: BTreeSet<_> = nodes
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
    keys: Vec<BTreeSet<DnfClause>>,
    groups: Vec<BTreeSet<Node>>,
    group_order: Vec<usize>,
) -> (Vec<BTreeSet<DnfClause>>, Vec<BTreeSet<Node>>) {
    let mut groups_out = vec![BTreeSet::new(); group_order.len()];
    let mut keys_out = vec![BTreeSet::new(); group_order.len()];
    for (i, (d, g)) in keys.into_iter().zip(groups.into_iter()).enumerate() {
        let j = group_order[i];
        keys_out[j] = d;
        groups_out[j] = g;
    }
    (keys_out, groups_out)
}

fn sort_nodes(ctx: &Context, group: BTreeSet<Node>) -> Vec<Node> {
    let mut parent_count: BTreeMap<Node, usize> = BTreeMap::new();

    // Find starting roots for this cluster, which are defined as any
    // node that's not an input to other nodes in the cluster.
    let children: BTreeSet<Node> = group
        .iter()
        .flat_map(|n| ctx.get_op(*n).unwrap().iter_children())
        .filter(|n| group.contains(n))
        .collect();

    let start: Vec<Node> = group
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
            .filter(|n| group.contains(n))
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
            .filter(|n| group.contains(n))
        {
            todo.push(child);
            *parent_count.get_mut(&child).unwrap() -= 1;
        }
        ordered_nodes.push(n);
    }
    ordered_nodes
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
                    // All other operations invalidate the reg-mem binding
                    reg_mem.remove(&op.out_reg().unwrap());
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
    for group in group_tapes.iter() {
        let mut active = BTreeSet::new();
        let mut out = vec![];
        for op in group {
            if let op::Op::Load { reg, .. } = op {
                if !active.contains(reg) {
                    continue; // skip this Load
                }
            } else {
                active.remove(&op.out_reg().unwrap());
                active.extend(op.input_reg_iter());
            }
            out.push(*op);
        }
        out_groups.push(out);
    }
    out_groups
}

/// Eliminates any `Store` operation which does not have a matching Load
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

pub fn buildy<F: Family>(
    ctx: &Context,
    root: Node,
    inline: usize,
) -> Result<tape::Tape<F>, Error> {
    if let Some(c) = ctx.const_value(root).unwrap() {
        let t = tape::ChoiceTape {
            tape: vec![op::Op::CopyImm {
                out: 0,
                imm: c as f32,
            }],
            choices: vec![],
        };
        return Ok(tape::Tape::new(vec![t], BTreeMap::new()));
    }

    let inline = pick_inline(ctx, root, inline);

    let (keys, mut groups) = find_groups(ctx, root, &inline);

    // For each DNF, add all of the inlined nodes
    for group in groups.iter_mut() {
        insert_inline_nodes(ctx, &inline, group);
    }
    let groups = groups; // remove mutability

    // Compute all of the global nodes
    let globals = {
        let mut globals: BTreeSet<Node> =
            groups.iter().flat_map(|g| find_globals(ctx, g)).collect();

        // Sanity-checking our globals and inlined variables
        assert!(globals.intersection(&inline).count() == 0);

        // The root is considered a global (TODO is this required?)
        assert!(!globals.contains(&root));
        globals.insert(root);
        globals
    };

    // Reorder the groups from root-to-leaf order
    //
    // Note that the nodes within each group remain unordered
    let group_order = compute_group_order(ctx, root, &groups, &globals);
    let (keys, groups) = apply_group_order(keys, groups, group_order);

    // Build a map from nodes-used-as-choices to indices in a flat list
    let choice_id: BTreeMap<Node, usize> = keys
        .iter()
        .flatten()
        .map(|v| v.root)
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    // Perform node ordering within each group
    let groups: Vec<Vec<Node>> =
        groups.into_iter().map(|g| sort_nodes(ctx, g)).collect();

    let mut group_tapes = vec![];
    let mut alloc = alloc::RegisterAllocator::new(ctx, F::REG_LIMIT);
    alloc.bind(&[(root, alloc::Allocation::Register(0))]);
    for group in groups.iter() {
        for node in group.iter() {
            alloc.op(*node, choice_id.get(node).copied());
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
    ] {
        group_tapes = pass(&group_tapes);
    }

    assert_eq!(group_tapes.len(), keys.len());
    let gt = group_tapes
        .into_iter()
        .zip(keys.into_iter())
        .map(|(g, k)| tape::ChoiceTape {
            tape: g,
            choices: k
                .into_iter()
                .map(|d| (choice_id[&d.root], d.choice))
                .collect(),
        })
        .collect();
    Ok(tape::Tape::new(gt, alloc.var_names()))
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
}
