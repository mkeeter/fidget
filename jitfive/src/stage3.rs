use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write;

use crate::error::Error;
use crate::indexed::{IndexMap, IndexVec};
use crate::stage0::{NodeIndex, VarIndex};
use crate::stage1::{GroupIndex, Source, TaggedOp};
use crate::stage2::Stage2;

/// A group represents a set of nodes which are enabled by the same set
/// of choices at `min` or `max` nodes.
///
/// This `Group` (unlike [`crate::stage1::Group`]) includes graph connections to
/// upstream and downstream groups.
///
/// In addition, unlike [`crate::stage2::Group`], it includes **tree**
/// connections to parent and child groups.  The tree parent is based on the
/// least common ancestor of all upstream groups.
#[derive(Default, Debug)]
pub struct Group {
    /// Choices which enable this group of nodes.
    ///
    /// If any choice in this array is valid, then the nodes of the group are
    /// enabled.  Choices are expressed in the positive form ("if choice _i_
    /// is *Left*, then the group is enabled").
    ///
    /// This array is expected to be sorted and unique, since it is used
    /// as a key when collecting nodes into groups.
    pub choices: Vec<Source>,

    /// Nodes in this group, in arbitrary order
    ///
    /// Indexes refer to nodes in the parent stage's `ops` array
    pub nodes: Vec<NodeIndex>,

    /// Downstream groups are farther from the root of the tree
    pub downstream: Vec<GroupIndex>,

    /// Upstream groups are closer to the root of the tree
    pub upstream: Vec<GroupIndex>,

    /// Parent of this group in the group tree, based on lowest common ancestor
    /// of the upstream nodes.
    pub parent: Option<GroupIndex>,

    /// Children of this group in the group tree; the opposite of `parent`
    pub children: Vec<GroupIndex>,
}

/// Stores a graph of math expressions, a graph of node groups, and a tree of
/// node groups (based on lowest common ancestors in the graph).
#[derive(Debug)]
pub struct Stage3 {
    /// Math operations, stored in arbitrary order and associated with a group
    pub ops: IndexVec<TaggedOp, NodeIndex>,

    /// Root of the tree
    pub root: NodeIndex,

    /// Groups of nodes and group graph links, stored in arbitrary order
    pub groups: IndexVec<Group, GroupIndex>,

    /// Number of nodes in the tree which make LHS/RHS choices
    pub num_choices: usize,

    /// Bi-directional map of variable names to indexes
    pub vars: IndexMap<String, VarIndex>,
}

impl From<&Stage2> for Stage3 {
    fn from(t: &Stage2) -> Self {
        let common_ancestors: IndexVec<BTreeSet<GroupIndex>, GroupIndex> = {
            let mut common_ancestors = IndexVec::new();
            common_ancestors.resize(t.groups.len(), None);
            for g in 0..t.groups.len() {
                populate_common_ancestors(
                    t,
                    GroupIndex::from(g),
                    &mut common_ancestors,
                );
            }
            common_ancestors.into_iter().map(Option::unwrap).collect()
        };

        let ranks: IndexVec<usize, GroupIndex> = {
            let mut ranks = IndexVec::new();
            ranks.resize(t.groups.len(), None);
            let root_group_index = t.ops[t.root].group;
            populate_ranks(t, root_group_index, 0, &mut ranks);
            ranks.into_iter().map(Option::unwrap).collect()
        };

        let parents: IndexVec<Option<GroupIndex>, GroupIndex> = t
            .groups
            .enumerate()
            .map(|(i, group)| {
                // Special case for the root of the tree
                if group.upstream.is_empty() {
                    return None;
                }
                let out = common_ancestors[i]
                    .iter()
                    .filter(|g| **g != i)
                    .max_by_key(|g| ranks[**g]);
                assert!(out.is_some());
                out.cloned()
            })
            .collect();

        let mut children: IndexVec<BTreeSet<GroupIndex>, GroupIndex> =
            IndexVec::new();
        children.resize_with(t.groups.len(), BTreeSet::new);
        for (g, p) in parents.enumerate() {
            if let Some(p) = *p {
                children[p].insert(g);
            }
        }

        let new_groups = children
            .into_iter()
            .map(|c| c.into_iter().collect()) // BTreeSet -> Vec
            .zip(parents.into_iter())
            .zip(t.groups.iter())
            .map(|((children, parent), group)| Group {
                choices: group.choices.clone(),
                nodes: group.nodes.clone(),
                upstream: group.upstream.clone(),
                downstream: group.downstream.clone(),
                parent,
                children,
            })
            .collect();

        Stage3 {
            ops: t.ops.clone(),
            root: t.root,
            groups: new_groups,
            num_choices: t.num_choices,
            vars: t.vars.clone(),
        }
    }
}

impl Stage3 {
    pub fn write_dot<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        writeln!(w, "digraph mygraph {{")?;
        writeln!(w, "compound=true")?;

        let root_group_index = self.ops[self.root].group;
        self.write_dot_recursive(w, root_group_index)?;

        // Write node edges afterwards
        for (i, op) in self.ops.enumerate() {
            for c in op.op.iter_children() {
                let alpha = if self.ops[c].group == op.group {
                    "FF"
                } else {
                    "40"
                };
                op.op.write_dot_edge(w, i, c, alpha)?;
            }
        }
        // Write group edges
        for (i, group) in self.groups.enumerate() {
            for c in &group.downstream {
                writeln!(
                    w,
                    "{} -> {} [ltail=cluster_{}, lhead=cluster_{}];",
                    if group.nodes.len() > 1 {
                        format!("SOURCE_{}", usize::from(i))
                    } else {
                        format!("n{}", usize::from(group.nodes[0]))
                    },
                    if self.groups[*c].nodes.len() > 1 {
                        format!("SINK_{}", usize::from(*c))
                    } else {
                        format!("n{}", usize::from(self.groups[*c].nodes[0]))
                    },
                    usize::from(i),
                    usize::from(*c)
                )?;
            }
        }
        writeln!(w, "}}")?;
        Ok(())
    }

    fn write_dot_recursive<W: Write>(
        &self,
        w: &mut W,
        i: GroupIndex,
    ) -> Result<(), Error> {
        writeln!(w, "subgraph cluster_{}_g {{", usize::from(i))?;
        writeln!(w, r#"color="grey""#)?;

        // This group's nodes live in their own cluster
        writeln!(w, "subgraph cluster_{} {{", usize::from(i))?;
        writeln!(w, r#"color="black""#)?;
        let group = &self.groups[i];
        for n in &group.nodes {
            let op = self.ops[*n].op;
            op.write_dot(w, *n, &self.vars)?;
        }
        // Invisible nodes to be used as group handles
        let i = usize::from(i);
        if group.nodes.len() > 1 {
            writeln!(w, "SINK_{} [shape=point style=invis]", i)?;
            writeln!(w, "SOURCE_{} [shape=point style=invis]", i)?;
            writeln!(w, "{{ rank = max; SOURCE_{} }}", i)?;
            writeln!(w, "{{ rank = min; SINK_{} }}", i)?;
        }
        writeln!(w, "}}")?;

        for child in &group.children {
            self.write_dot_recursive(w, *child)?;
        }
        writeln!(w, "}}")?;
        Ok(())
    }
}

fn populate_common_ancestors(
    t: &Stage2,
    g: GroupIndex,
    out: &mut IndexVec<Option<BTreeSet<GroupIndex>>, GroupIndex>,
) {
    // Skip if we've already handled this node
    if out[g].is_some() {
        return;
    }
    let group = &t.groups[g];
    let mut common_count: BTreeMap<_, usize> = BTreeMap::new();
    for upstream_group in &group.upstream {
        // Recurse upwards towards the root
        populate_common_ancestors(t, *upstream_group, out);
        for a in out[*upstream_group].as_ref().unwrap() {
            *common_count.entry(*a).or_default() += 1;
        }
    }

    // Common ancestors are reachable from every upstream group of the target,
    // which we check by counting how many upstream groups can reach each
    // potential common ancestor.
    let mut s: BTreeSet<GroupIndex> = common_count
        .iter()
        .filter(|(_g, c)| **c == group.upstream.len())
        .map(|(g, _c)| *g)
        .collect();

    // Every node is its own common ancestor
    s.insert(g);

    out[g] = Some(s);
}

fn populate_ranks(
    t: &Stage2,
    g: GroupIndex,
    rank: usize,
    out: &mut IndexVec<Option<usize>, GroupIndex>,
) {
    // Breadth-first search!
    let mut todo = VecDeque::new();
    todo.push_back((g, rank));
    while let Some((g, rank)) = todo.pop_front() {
        if let Some(r) = out[g] {
            // Nothing to do here, other than a sanity-check
            assert!(r <= rank);
        } else {
            for downstream_group in &t.groups[g].downstream {
                todo.push_back((*downstream_group, rank + 1));
            }
            out[g] = Some(rank);
        }
    }
}
