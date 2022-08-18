//! Passes to build the final `Compiler` object
//!
//! We expect passes to be run in the following order:
//! - `flatten_tree`
//! - `find_groups`
//! - `group_graph`
//! - `group_tree`
//! - `group_weight`
//! - `sort_groups`
//! - `sort_nodes`
//! - `node_lifetime`

pub(crate) mod find_groups;
pub(crate) mod flatten_tree;
pub(crate) mod group_graph;
pub(crate) mod group_tree;
pub(crate) mod group_weight;
pub(crate) mod node_lifetime;
pub(crate) mod sort_groups;
pub(crate) mod sort_nodes;
