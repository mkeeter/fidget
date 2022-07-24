//! Container types with strongly-typed indexes.
use crate::error::Error;
use std::collections::HashMap;

/// Stores a set of `(Value, Index)` tuples, with lookup in both directions.
///
/// Implemented using a `Vec<Value>` and a `HashMap<Value, Index>`.
///
/// The `Index` type should be a wrapper around a `usize` and be convertible
/// in both directions; it is typically passed around using `Copy`.  A suitable
/// index type can be constructed with [define_index].
///
/// The `Value` type may be larger and is passed around by reference. However,
/// it must be `Clone`, because it is stored twice in the data structure (once
/// in the `Vec` and once in the `HashMap`).
#[derive(Debug)]
pub struct IndexMap<Value, Index> {
    data: Vec<Value>,
    map: HashMap<Value, Index>,
}

impl<Value, Index> Default for IndexMap<Value, Index> {
    fn default() -> Self {
        Self {
            data: vec![],
            map: HashMap::new(),
        }
    }
}

impl<Value, Index> IndexMap<Value, Index>
where
    Value: Eq + std::hash::Hash + Clone,
    Index: Eq + std::hash::Hash + Copy + From<usize>,
    usize: From<Index>,
{
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn get_by_index(&self, v: Index) -> Option<&Value> {
        self.data.get(usize::from(v))
    }
    /// Insert the given value into the map, returning a handle.
    ///
    /// If the value is already in the map, the handle will be to the existing
    /// instance (so it will not be inserted twice).
    pub fn insert(&mut self, v: Value) -> Index {
        *self.map.entry(v.clone()).or_insert_with(|| {
            let out = Index::from(self.data.len());
            self.data.push(v);
            out
        })
    }
    /// Removes the last value stored in the container.
    ///
    /// This is _usually_ the most recently inserted value, except when
    /// `insert` is called on a duplicate.
    pub fn pop(&mut self) -> Result<Value, Error> {
        match self.data.pop() {
            Some(v) => {
                self.map.remove(&v);
                Ok(v)
            }
            None => Err(Error::EmptyMap),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Value, Index)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (v, Index::from(i)))
    }
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.data.iter()
    }
    pub fn keys(&self) -> impl Iterator<Item = Index> {
        (0..self.data.len()).map(Index::from)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A `Vec<Value>` with strongly-typed indexes, used to improve the type-safety
/// of data storage.
///
/// The `Index` type should be a wrapper around a `usize` and be convertible
/// in both directions; it is typically passed around using `Copy`.  A suitable
/// index type can be constructed with [define_index].
#[derive(Debug)]
pub struct IndexVec<Value, Index> {
    data: Vec<Value>,
    _phantom: std::marker::PhantomData<*const Index>,
}

impl<Value, Index> IndexVec<Value, Index> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.data.iter()
    }
}

impl<Value, Index> std::ops::Index<Index> for IndexVec<Value, Index>
where
    usize: From<Index>,
{
    type Output = Value;
    fn index(&self, i: Index) -> &Value {
        &self.data[usize::from(i)]
    }
}

impl<Value, Index> std::ops::IndexMut<Index> for IndexVec<Value, Index>
where
    usize: From<Index>,
{
    fn index_mut(&mut self, i: Index) -> &mut Value {
        &mut self.data[usize::from(i)]
    }
}

impl<Value, Index> From<Vec<Value>> for IndexVec<Value, Index> {
    fn from(data: Vec<Value>) -> Self {
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
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct $name(usize);
        impl From<usize> for $name {
            fn from(v: usize) -> Self {
                Self(v)
            }
        }
        impl From<$name> for usize {
            fn from(v: $name) -> Self {
                v.0
            }
        }
    };
}
pub(crate) use define_index;
