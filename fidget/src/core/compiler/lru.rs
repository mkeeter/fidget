/// Single node in the doubly-linked list
#[derive(Copy, Clone, Default)]
struct LruNode {
    prev: u8,
    next: u8,
}

/// Dead-simple LRU cache, implemented as a doubly-linked list with a static
/// backing array.
///
/// ```text
///              <-- prev next -->
///      -------      -------      -------      --------
///      |  a  | <--> |  b  | <--> |  c  | <--> | head | <--|
///      -------      -------      -------      --------    |
///         ^                       oldest       newest     |
///         |-----------------------------------------------|
/// ```
pub struct Lru<const N: usize> {
    data: [LruNode; N],
    head: u8,
}

impl<const N: usize> Lru<N> {
    pub fn new() -> Self {
        let mut out = Self {
            data: [LruNode::default(); N],
            head: 0,
        };
        for i in 0..N {
            out.data[i as usize].next = ((i + 1) % N) as u8;
            out.data[i as usize].prev =
                (i.checked_sub(1).unwrap_or(N - 1)) as u8;
        }
        out
    }

    /// Remove a node from the linked list
    #[inline]
    fn remove(&mut self, i: u8) {
        let node = self.data[i as usize];
        self.data[node.prev as usize].next = self.data[i as usize].next;
        self.data[node.next as usize].prev = self.data[i as usize].prev;
    }

    /// Inserts node `i` before location `next`
    #[inline]
    fn insert_before(&mut self, i: u8, next: u8) {
        let prev = self.data[next as usize].prev;
        self.data[prev as usize].next = i;
        self.data[next as usize].prev = i;
        self.data[i as usize] = LruNode { next, prev };
    }

    /// Mark the given node as newest
    #[inline]
    pub fn poke(&mut self, i: u8) {
        let prev_newest = self.head;
        if prev_newest == i {
            return;
        } else if self.data[prev_newest as usize].prev != i {
            // If this wasn't the oldest node, then remove it and reinsert it
            // right before the head of the list.
            self.remove(i);
            self.insert_before(i, self.head);
        }
        self.head = i; // rotate the head back by one
    }

    /// Look up the oldest node in the list, marking it as newest
    #[inline]
    pub fn pop(&mut self) -> u8 {
        let out = self.data[self.head as usize].prev;
        self.head = out; // rotate
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tiny_lru() {
        let mut lru: Lru<2> = Lru::new();
        lru.poke(0);
        assert!(lru.pop() == 1);
        assert!(lru.pop() == 0);

        lru.poke(1);
        assert!(lru.pop() == 0);
        assert!(lru.pop() == 1);
    }

    #[test]
    fn test_medium_lru() {
        let mut lru: Lru<10> = Lru::new();
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
