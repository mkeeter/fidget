//! Input variables to math expressions
//!
//! A [`Var`] maintains a persistent identity from
//! [`Tree`](crate::context::Tree) to [`Context`](crate::context::Node) (where
//! it is wrapped in a [`Op::Input`](crate::context::Op::Input)) to evaluation
//! (where [`Function::vars`](crate::eval::Function::vars) maps from `Var` to
//! index in the argument list).
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
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum Var {
    X,
    Y,
    Z,
    V(VarIndex),
}

/// Type for a variable index (implemented as a `u64`)
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

/// Map from [`Var`] to a particular value
///
/// This is equivalent to a
/// [`HashMap<Var, T>`](std::collections::HashMap) and as such does not include
/// per-function documentation.
///
/// The advantage over a `HashMap` is that for common variables (`X`, `Y`, `Z`),
/// no allocation is required.
#[derive(Serialize, Deserialize)]
pub struct VarMap<T> {
    x: Option<T>,
    y: Option<T>,
    z: Option<T>,
    v: HashMap<VarIndex, T>,
}

impl<T> Default for VarMap<T> {
    fn default() -> Self {
        Self {
            x: None,
            y: None,
            z: None,
            v: HashMap::default(),
        }
    }
}

#[allow(missing_docs)]
impl<T> VarMap<T> {
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
    pub fn get(&self, v: &Var) -> Option<&T> {
        match v {
            Var::X => self.x.as_ref(),
            Var::Y => self.y.as_ref(),
            Var::Z => self.z.as_ref(),
            Var::V(v) => self.v.get(v),
        }
    }

    pub fn get_mut(&mut self, v: &Var) -> Option<&mut T> {
        match v {
            Var::X => self.x.as_mut(),
            Var::Y => self.y.as_mut(),
            Var::Z => self.z.as_mut(),
            Var::V(v) => self.v.get_mut(v),
        }
    }

    pub fn entry(&mut self, v: Var) -> VarMapEntry<T> {
        match v {
            Var::X => VarMapEntry::Option(&mut self.x),
            Var::Y => VarMapEntry::Option(&mut self.y),
            Var::Z => VarMapEntry::Option(&mut self.z),
            Var::V(v) => VarMapEntry::Hash(self.v.entry(v)),
        }
    }
}

/// Entry into a [`VarMap`]; equivalent to [`std::collections::hash_map::Entry`]
///
/// The implementation has just enough functions to be useful; if you find
/// yourself wanting the rest of the entry API, it could easily be expanded.
#[allow(missing_docs)]
pub enum VarMapEntry<'a, T> {
    Option(&'a mut Option<T>),
    Hash(std::collections::hash_map::Entry<'a, VarIndex, T>),
}

#[allow(missing_docs)]
impl<'a, T> VarMapEntry<'a, T> {
    pub fn or_insert(self, default: T) -> &'a mut T {
        match self {
            VarMapEntry::Option(o) => match o {
                Some(v) => v,
                None => {
                    *o = Some(default);
                    o.as_mut().unwrap()
                }
            },
            VarMapEntry::Hash(e) => e.or_insert(default),
        }
    }
}

impl<T> std::ops::Index<&Var> for VarMap<T> {
    type Output = T;
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
        let p = m.entry(v).or_insert(123);
        assert_eq!(*p, 123);
        let p = m.entry(v).or_insert(456);
        assert_eq!(*p, 123);
    }
}
