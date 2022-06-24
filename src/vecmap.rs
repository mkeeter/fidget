use crate::Error;
use std::collections::HashMap;

/// Stores a set of `(Value, Index)` tuples, with lookup in both directions.
///
/// Implemented using a `Vec<Value>` and a `HashMap<Value, Index>`.
///
/// The `Index` type should be a wrapper around a `usize` and be convertible
/// in both directions; it is typically passed around using `Copy`.
///
/// The `Value` type may be larger and is passed around by reference. However,
/// it must be `Clone`, because it is stored twice in the data structure (once
/// in the `Vec` and once in the `HashMap`).
#[derive(Debug)]
pub struct VecMap<Value, Index> {
    data: Vec<Value>,
    map: HashMap<Value, Index>,
}

impl<Value, Index> Default for VecMap<Value, Index> {
    fn default() -> Self {
        Self {
            data: vec![],
            map: HashMap::new(),
        }
    }
}

impl<Value, Index> VecMap<Value, Index>
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
    pub fn get_by_value(&self, d: &Value) -> Option<Index> {
        self.map.get(d).cloned()
    }
    pub fn get_by_index(&self, v: Index) -> Option<&Value> {
        self.data.get(usize::from(v))
    }
    pub fn insert(&mut self, v: Value) -> Index {
        let out = Index::from(self.data.len());
        self.data.push(v.clone());
        self.map.insert(v, out);
        out
    }
    pub fn pop(&mut self) -> Result<Value, Error> {
        match self.data.pop() {
            Some(v) => {
                self.map.remove(&v);
                Ok(v)
            }
            None => Err(Error::EmptyMap),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Index)> {
        self.map.iter()
    }
}
