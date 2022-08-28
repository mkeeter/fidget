use std::collections::{btree_map::Entry, BTreeMap, BTreeSet};

/// Simple generic priority queue
pub struct PriorityQueue<K, V> {
    data: BTreeMap<V, K>,
    queue: BTreeSet<(K, V)>,
}

impl<K, V> Default for PriorityQueue<K, V> {
    fn default() -> Self {
        Self {
            data: BTreeMap::new(),
            queue: BTreeSet::new(),
        }
    }
}
impl<K, V> PriorityQueue<K, V>
where
    V: Copy + Ord,
    K: Copy + Ord,
{
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn insert_or_update(&mut self, value: V, priority: K) {
        match self.data.entry(value) {
            Entry::Vacant(e) => {
                e.insert(priority);
            }
            Entry::Occupied(mut e) => {
                self.queue.remove(&(*e.get(), value));
                *e.get_mut() = priority;
            }
        };
        self.queue.insert((priority, value));
    }
    /// Removes the given value, returning its priority
    pub fn remove(&mut self, value: V) -> Option<K> {
        if let Some(priority) = self.data.remove(&value) {
            self.queue.remove(&(priority, value));
            Some(priority)
        } else {
            None
        }
    }
    pub fn pop(&mut self) -> Option<V> {
        // This could be cleaner once #62924 map_first_last is stabilized
        if let Some((priority, value)) = self.queue.iter().next().cloned() {
            self.queue.remove(&(priority, value));
            self.data.remove(&value);
            Some(value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue() {
        let a = "hello";
        let b = "world";
        let mut q = PriorityQueue::new();
        q.insert_or_update(a, 0);
        q.insert_or_update(b, 1);
        assert_eq!(q.pop(), Some("hello"));
        assert_eq!(q.pop(), Some("world"));

        let mut q = PriorityQueue::new();
        q.insert_or_update(a, 0);
        q.insert_or_update(b, 1);
        q.insert_or_update(a, 2);
        assert_eq!(q.pop(), Some("world"));
        assert_eq!(q.pop(), Some("hello"));
    }
}
