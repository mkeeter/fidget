/// Single node in the doubly-linked list
#[derive(Copy, Clone)]
struct LruNode {
    prev: usize,
    next: usize,
}

/// Dead-simple LRU cache, implemented as a doubly-linked list with a static
/// backing array.
///
/// ```text
///              <-- prev next -->
///      -------      -------      -------      --------
///      |  a  | <--> |  b  | <--> |  c  | <--> | size | <--|
///      -------      -------      -------      --------    |
///         ^                       oldest                  |
///         |-----------------------------------------------|
/// ```
pub struct Lru<const S: usize> {
    data: [LruNode; S],
    size: usize,
}

impl<const S: usize> Lru<S> {
    pub fn new(size: usize) -> Self {
        // We store one extra node to represent the head of the array, so the
        // backing array must be larger than size
        assert!(S > size);
        let mut data = [LruNode { prev: 0, next: 0 }; S];
        for i in 0..size {
            data[i].next = i + 1;
            data[i + 1].prev = i;
        }
        // data[size] is the head of the tape
        data[0].prev = size;
        data[size].next = 0;

        Self { data, size }
    }

    /// Mark the given node as newest
    pub fn poke(&mut self, i: usize) {
        assert!(i < self.size);
        let prev_newest = self.data[self.size].next;
        if prev_newest != i {
            // Remove this node from the list
            self.data[self.data[i].prev].next = self.data[i].next;
            self.data[self.data[i].next].prev = self.data[i].prev;

            // Reinsert the node between prev_newest and the head
            self.data[prev_newest].prev = i;
            self.data[self.size].next = i;
            self.data[i].next = prev_newest;
            self.data[i].prev = self.size;
        }
    }
    /// Look up the oldest node in the list, marking it as newest
    pub fn pop(&mut self) -> usize {
        let out = self.data[self.size].prev;
        self.poke(out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_lru() {
        let mut lru: Lru<3> = Lru::new(2);
        lru.poke(0);
        assert!(lru.pop() == 1);
        assert!(lru.pop() == 0);

        lru.poke(1);
        assert!(lru.pop() == 0);
        assert!(lru.pop() == 1);
    }

    #[test]
    fn test_medium_lru() {
        let mut lru: Lru<11> = Lru::new(10);
        lru.poke(0);
        for _ in 0..9 {
            assert!(lru.pop() != 0);
        }
        assert!(lru.pop() == 0);

        lru.poke(1);
        for _ in 0..9 {
            assert!(lru.pop() != 1);
        }
        assert!(lru.pop() == 1);

        lru.poke(4);
        lru.poke(5);
        for _ in 0..8 {
            assert!(!matches!(lru.pop(), 4 | 5));
        }
        assert!(lru.pop() == 4);
        assert!(lru.pop() == 5);
    }
}
