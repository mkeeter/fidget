//! Input variables to math expressions
//!
//! A [`Var`] maintains a persistent identity from
//! [`Tree`](crate::context::Tree) to [`Context`](crate::context::Node) (where
//! it is wrapped in a [`Op::Input`](crate::context::Op::Input)) to evaluation
//! (where [`Tape::vars`](crate::eval::Tape::vars) maps from `Var` to index in
//! the argument list).
use crate::context::{Context, IntoNode, Node};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The [`Var`] type is an input to a math expression
///
/// We pre-define common variables (e.g. `X`, `Y`, `Z`) but also allow for fully
/// customized values (using [`Var::V`]).
///
/// Variables are "global", in that every instance of `Var::X` represents the
/// same thing.  To generate a "local" variable, [`Var::new`] picks a random
/// 64-bit value, which is very unlikely to collide with anything else.
#[derive(
    Copy,
    Clone,
    Debug,
    Hash,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Serialize,
    Deserialize,
)]
pub enum Var {
    /// Variable representing the X axis for 2D / 3D shapes
    X,
    /// Variable representing the Y axis for 2D / 3D shapes
    Y,
    /// Variable representing the Z axis for 3D shapes
    Z,
    /// Generic variable
    V(VarIndex),
}

/// Type for a variable index (implemented as a `u64`), used in [`Var::V`]
#[derive(
    Copy,
    Clone,
    Debug,
    Hash,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
pub struct VarIndex(u64);

impl Var {
    /// Returns a new variable, with a random 64-bit index
    ///
    /// The odds of collision with any previous variable are infintesimally
    /// small; if you are generating billions of random variables, something
    /// else in the system is likely to break before collisions become an issue.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let v: u64 = rand::random();
        Var::V(VarIndex(v))
    }

    /// Returns the [`VarIndex`] from a [`Var::V`] instance, or `None`
    pub fn index(&self) -> Option<VarIndex> {
        if let Var::V(i) = *self {
            Some(i)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Var::X => write!(f, "X"),
            Var::Y => write!(f, "Y"),
            Var::Z => write!(f, "Z"),
            Var::V(VarIndex(v)) if *v < 256 => write!(f, "v_{v}"),
            Var::V(VarIndex(v)) => write!(f, "V({v:x})"),
        }
    }
}

impl IntoNode for Var {
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error> {
        Ok(ctx.var(self))
    }
}

/// Map from [`Var`] to a particular index
///
/// Variable indexes are automatically assigned the first time
/// [`VarMap::insert`] is called on that variable.
///
/// Indexes are guaranteed to be tightly packed, i.e. a map `vars` will contains
/// values from `0..vars.len()`.
///
/// For efficiency, this type does not allocate heap memory for `Var::X/Y/Z`.
#[derive(Default, Serialize, Deserialize)]
pub struct VarMap {
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
    v: HashMap<VarIndex, usize>,
}

#[allow(missing_docs)]
impl VarMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.x.is_some() as usize
            + self.y.is_some() as usize
            + self.z.is_some() as usize
            + self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_none()
            && self.y.is_none()
            && self.z.is_none()
            && self.v.is_empty()
    }
    pub fn get(&self, v: &Var) -> Option<usize> {
        match v {
            Var::X => self.x,
            Var::Y => self.y,
            Var::Z => self.z,
            Var::V(v) => self.v.get(v).cloned(),
        }
    }
    /// Inserts a variable if not already present in the map
    ///
    /// The index is automatically assigned.
    pub fn insert(&mut self, v: Var) {
        let next = self.len();
        match v {
            Var::X => self.x.get_or_insert(next),
            Var::Y => self.y.get_or_insert(next),
            Var::Z => self.z.get_or_insert(next),
            Var::V(v) => self.v.entry(v).or_insert(next),
        };
    }

    pub(crate) fn check_tracing_arguments<T>(
        &self,
        vars: &[T],
    ) -> Result<(), Error> {
        if vars.len() < self.len() {
            // It's okay to be passed extra vars, because expressions may have
            // been simplified.
            Err(Error::BadVarSlice(vars.len(), self.len()))
        } else {
            Ok(())
        }
    }

    pub(crate) fn check_bulk_arguments<T, V: std::ops::Deref<Target = [T]>>(
        &self,
        vars: &[V],
    ) -> Result<(), Error> {
        // It's fine if the caller has given us extra variables (e.g. due to
        // tape simplification), but it must have given us enough.
        if vars.len() < self.len() {
            Err(Error::BadVarSlice(vars.len(), self.len()))
        } else {
            let Some(n) = vars.first().map(|v| v.len()) else {
                return Ok(());
            };
            if vars.iter().any(|v| v.len() == n) {
                Ok(())
            } else {
                Err(Error::MismatchedSlices)
            }
        }
    }
}

impl std::ops::Index<&Var> for VarMap {
    type Output = usize;
    fn index(&self, v: &Var) -> &Self::Output {
        match v {
            Var::X => self.x.as_ref().unwrap(),
            Var::Y => self.y.as_ref().unwrap(),
            Var::Z => self.z.as_ref().unwrap(),
            Var::V(v) => &self.v[v],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn var_identity() {
        let v1 = Var::new();
        let v2 = Var::new();
        assert_ne!(v1, v2);
    }

    #[test]
    fn var_map() {
        let v = Var::new();
        let mut m = VarMap::new();
        assert!(m.get(&v).is_none());
        m.insert(v);
        assert_eq!(m.get(&v), Some(0));
        m.insert(v);
        assert_eq!(m.get(&v), Some(0));

        let u = Var::new();
        assert!(m.get(&u).is_none());
        m.insert(u);
        assert_eq!(m.get(&u), Some(1));
    }
}
