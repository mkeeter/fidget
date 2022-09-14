//! Container types with strongly-typed indexes.
use crate::error::Error;
use std::collections::HashMap;

/// Stores a set of `(V, I)` tuples, with lookup in both directions.
///
/// Implemented using a `Vec<V>` and a `HashMap<V, I>`.
///
/// The index type `I` should be a wrapper around a `usize` and be convertible
/// in both directions using the `Index` trait; it is typically passed around
/// using `Copy`.  A suitable index type can be constructed with [define_index].
///
/// The `V` type may be larger and is passed around by reference. However,
/// it must be `Clone`, because it is stored twice in the data structure (once
/// in the `Vec` and once in the `HashMap`).
#[derive(Clone, Debug)]
pub(crate) struct IndexMap<V, Index> {
    data: Vec<V>,
    map: HashMap<V, Index>,
}

impl<V, Index> Default for IndexMap<V, Index> {
    fn default() -> Self {
        Self {
            data: vec![],
            map: HashMap::new(),
        }
    }
}

pub(crate) trait Index {
    fn new(i: usize) -> Self;
    fn get(&self) -> usize;
}

impl<V, I> IndexMap<V, I>
where
    V: Eq + std::hash::Hash + Clone,
    I: Eq + std::hash::Hash + Copy + Index,
{
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn get_by_index(&self, i: I) -> Option<&V> {
        self.data.get(i.get())
    }
    /// Insert the given value into the map, returning a handle.
    ///
    /// If the value is already in the map, the handle will be to the existing
    /// instance (so it will not be inserted twice).
    pub fn insert(&mut self, v: V) -> I {
        *self.map.entry(v.clone()).or_insert_with(|| {
            let out = I::new(self.data.len());
            self.data.push(v);
            out
        })
    }

    /// Removes the last value stored in the container.
    ///
    /// This is _usually_ the most recently inserted value, except when
    /// `insert` is called on a duplicate.
    pub fn pop(&mut self) -> Result<V, Error> {
        match self.data.pop() {
            Some(v) => {
                self.map.remove(&v);
                Ok(v)
            }
            None => Err(Error::EmptyMap),
        }
    }
    pub fn keys(&self) -> impl Iterator<Item = I> {
        (0..self.data.len()).map(I::new)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A `Vec<V>` with strongly-typed indexes, used to improve the type-safety
/// of data storage.
///
/// The `Index` type should be a wrapper around a `usize` and be convertible
/// in both directions; it is typically passed around using `Copy`.  A suitable
/// index type can be constructed with [define_index].
#[derive(Clone, Debug)]
pub struct IndexVec<V, I> {
    data: Vec<V>,
    _phantom: std::marker::PhantomData<*const I>,
}

impl<V, I> Default for IndexVec<V, I> {
    fn default() -> Self {
        Self {
            data: vec![],
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<V, I> std::iter::IntoIterator for IndexVec<V, I> {
    type Item = V;
    type IntoIter = std::vec::IntoIter<V>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<V, I> FromIterator<V> for IndexVec<V, I> {
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<V, I> std::ops::Index<I> for IndexVec<V, I>
where
    I: Index,
{
    type Output = V;
    fn index(&self, i: I) -> &V {
        &self.data[i.get()]
    }
}

impl<V, I> std::ops::IndexMut<I> for IndexVec<V, I>
where
    I: Index,
{
    fn index_mut(&mut self, i: I) -> &mut V {
        &mut self.data[i.get()]
    }
}

impl<V, I> From<Vec<V>> for IndexVec<V, I> {
    fn from(data: Vec<V>) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Defines an index type suitable for use in an [`IndexMap`] or [`IndexVec`].
macro_rules! define_index {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Copy, Clone, Default, Debug, Eq, PartialEq, Hash, Ord, PartialOrd,
        )]
        pub struct $name(usize);
        impl crate::context::indexed::Index for $name {
            fn new(i: usize) -> Self {
                Self(i)
            }
            fn get(&self) -> usize {
                self.0
            }
        }
    };
}
pub(crate) use define_index;
