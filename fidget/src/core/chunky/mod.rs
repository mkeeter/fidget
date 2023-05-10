use crate::{
    context::{BinaryOpcode, Context, Node, Op},
    eval::tracing::Choice,
    Error,
};

mod alloc;
mod op;

use std::collections::{BTreeMap, BTreeSet};

/// Globally unique index for a particular choice node
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ChoiceIndex(usize);

/// Globally unique index for a particular DNF group
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct GroupIndex(usize);

/// Globally unique index for a pseudo-register
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct Register(usize);

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

fn find_groups(
    ctx: &Context,
    root: Node,
    inline: &BTreeSet<Node>,
) -> (Vec<Option<BTreeSet<DnfClause>>>, Vec<BTreeSet<Node>>) {
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
            dnf_nodes.entry(None)
        } else {
            dnf_nodes.entry(Some(d.into_iter().map(|v| v.unwrap()).collect()))
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
    //
    // At this point, a GroupIndex refers to order within the `keys` and
    // `groups` vectors.
    let global_node_to_group: BTreeMap<Node, GroupIndex> = groups
        .iter()
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
    for (g, nodes) in groups.iter().enumerate() {
        let g = GroupIndex(g);
        let map: BTreeSet<GroupIndex> = nodes
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
        group_order[g.0] = i;
    }
    group_order
}

fn apply_group_order(
    keys: Vec<Option<BTreeSet<DnfClause>>>,
    groups: Vec<BTreeSet<Node>>,
    group_order: Vec<usize>,
) -> (Vec<Option<BTreeSet<DnfClause>>>, Vec<BTreeSet<Node>>) {
    let mut groups_out = vec![BTreeSet::new(); group_order.len()];
    let mut keys_out = vec![None; group_order.len()];
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

/// Eliminates any Load operation which already has the value in the register
fn eliminate_forward_loads(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    // Records the active register -> memory mapping.  If a register is only
    // used as a local value, then this map does not contain its value.
    let mut reg_mem = BTreeMap::new();
    let mut out_groups = vec![];

    for group in group_tapes.iter().rev() {
        let mut out = vec![];
        for op in group.iter().rev() {
            match *op {
                op::Op::Load(reg, mem) => {
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
                op::Op::Store(reg, mem) => {
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

/// Eliminates any Load operation which does not use the register
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
            if let op::Op::Load(reg, ..) = op {
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

/// Eliminates any Store operation which does not have a matching Load
fn eliminate_reverse_stores(group_tapes: &[Vec<op::Op>]) -> Vec<Vec<op::Op>> {
    let mut out_groups = vec![];
    let mut active = BTreeSet::new();
    for group in group_tapes.iter() {
        let mut out = vec![];
        for op in group {
            if let op::Op::Load(_reg, mem) = op {
                active.insert(mem);
            } else if let op::Op::Store(_reg, mem) = op {
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

pub fn buildy(
    ctx: &Context,
    root: Node,
    inline: usize,
    registers: u8,
) -> Result<(), Error> {
    let start = std::time::Instant::now();

    let inline = pick_inline(ctx, root, inline);
    println!("got {} inline nodes", inline.len());

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
        .flat_map(|k| k.iter().flatten())
        .map(|v| v.root)
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    // Perform node ordering within each group
    let groups: Vec<Vec<Node>> =
        groups.into_iter().map(|g| sort_nodes(ctx, g)).collect();

    let mut group_tapes = vec![];
    let mut alloc = alloc::RegisterAllocator::new(ctx, registers);
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

    let pruned_groups = eliminate_forward_loads(&group_tapes);
    let pruned_groups2 = eliminate_reverse_loads(&pruned_groups);
    let pruned_groups3 = eliminate_reverse_stores(&pruned_groups2);

    println!("done in {:?}", start.elapsed());

    ////////////////////////////////////////////////////////////////////////////
    // Verbose logging and debug info
    for tape in &group_tapes {
        for op in tape {
            println!("{op:?}");
        }
        println!("--------");
    }
    println!("=======================\nPruned results");
    for tape in &pruned_groups {
        for op in tape {
            println!("{op:?}");
        }
        println!("--------");
    }
    println!("=======================\nPruned 2 results 2");
    for tape in &pruned_groups2 {
        for op in tape {
            println!("{op:?}");
        }
        println!("--------");
    }
    println!("=======================\nPruned 3 results 3");
    for tape in &pruned_groups3 {
        for op in tape {
            println!("{op:?}");
        }
        println!("--------");
    }

    for (group, dnf) in groups.iter().zip(&keys) {
        println!("=======================");
        println!("{dnf:?}");
        for n in group {
            print!("{n:?} = {:?}", ctx.get_op(*n).unwrap());
            if globals.contains(n) {
                print!(" (GLOBAL)");
            } else if inline.contains(n) {
                print!(" (INLINE)");
            }
            println!();
        }
    }

    println!("got {} DNF -> node clusters", keys.len());
    println!("got {} globals", globals.len());

    let mut hist: BTreeMap<_, usize> = BTreeMap::new();
    for d in &groups {
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
    for d in &groups {
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
        std::fs::write("out.dot", ctx.dot()).unwrap();
        buildy(&ctx, root, 9, 24).unwrap();
    }
}
