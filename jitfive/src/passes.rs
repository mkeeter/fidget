//! Passes to build the final `Compiler` object
//!
//! We expect passes to be run in the following order:
//! - `flatten_tree`
//! - `find_groups`
//! - `group_graph`
//! - `group_tree` (any time after `group_graph`)
//! - `sort_groups` (any time after `group_tree`)
//! - `sort_nodes` (any time after `group_tree`)
//! - `node_lifetime` (any time after `sort_nodes` and `sort_groups`)
//! - `group_weight` (any time after `group_tree`)

pub(crate) mod find_groups;
pub(crate) mod flatten_tree;
pub(crate) mod group_graph;
pub(crate) mod group_tree;
pub(crate) mod group_weight;
pub(crate) mod node_lifetime;
pub(crate) mod sort_groups;
pub(crate) mod sort_nodes;
