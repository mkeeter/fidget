use std::collections::BTreeMap;

pub struct Bimap<L, R> {
    left: BTreeMap<L, R>,
    right: BTreeMap<R, L>,
}

impl<L, R> Default for Bimap<L, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L, R> Bimap<L, R> {
    pub fn new() -> Self {
        Self {
            left: BTreeMap::new(),
            right: BTreeMap::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
    pub fn len(&self) -> usize {
        assert_eq!(self.left.len(), self.right.len());
        self.left.len()
    }
}

impl<L, R> Bimap<L, R>
where
    L: Ord + Copy,
    R: Ord + Copy,
{
    /// Inserts a new pair into the map.
    ///
    /// If the map already contained a pair associated with either of the given
    /// keys, returns `false`; otherwise, returns `true`
    pub fn insert(&mut self, left: L, right: R) -> bool {
        let mut new = true;
        new &= self.erase_left(&left).is_none();
        new &= self.erase_right(&right).is_none();
        self.left.insert(left, right);
        self.right.insert(right, left);
        new
    }

    pub fn get_left(&self, left: &L) -> Option<&R> {
        self.left.get(left)
    }
    pub fn get_right(&self, right: &R) -> Option<&L> {
        self.right.get(right)
    }
    pub fn erase_left(&mut self, left: &L) -> Option<R> {
        if let Some(r) = self.left.remove(left) {
            self.right.remove(&r).unwrap();
            Some(r)
        } else {
            None
        }
    }
    pub fn erase_right(&mut self, right: &R) -> Option<L> {
        if let Some(l) = self.right.remove(right) {
            self.left.remove(&l).unwrap();
            Some(l)
        } else {
            None
        }
    }
}
