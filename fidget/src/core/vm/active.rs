#[derive(Copy, Clone, Default)]
struct RegisterSet([u32; 8]);

#[derive(Clone)]
struct Node<S> {
    /// Bottom-up values, populated by pushing individual items
    accum: S,

    /// Top-down values, populated by updating ranges
    value: S,
}

trait Set {
    fn zero() -> Self;
    fn merge(&mut self, other: &Self);
}

impl Set for RegisterSet {
    fn zero() -> Self {
        Self([0; 8])
    }
    fn merge(&mut self, other: &RegisterSet) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= *b;
        }
    }
}

// Look upon my tree, ye mighty, and despair!
// (this is here for reference so I can check it when writing code below)
//
//                  -------------------------------
//                 /               |               \
//          ---------------        |        ---------------
//         /       |       \       |       /       |       \
//      -------    |    -------    |    -------    |    -------
//     /   |   \   |   /   |   \   |   /   |   \   |   /   |   \
//     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14      index
//     0       1       2       3       4       5       6       7      item index
//
//     0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1  LSB
//     0   0   1   1   0   0   1   1   0   0   1   1   0   0   1   1
//     0   0   0   0   1   1   1   1   0   0   0   0   1   1   1   1
//     0   0   0   0   0   0   0   0   1   1   1   1   1   1   1   1

/// Set of a bitfield-like type with efficient range queries and updates
///
/// Under the hood, this is implemented as a modified implicit in-order forest;
/// see [the original post](https://thume.ca/2021/03/14/iforests/) and
/// [this excellent writeup](https://github.com/havelessbemore/dastal/blob/main/src/segmentTree/inOrderSegmentTree.md)
/// for details.
#[derive(Clone, Default)]
struct RangeBitset<S>(Vec<Node<S>>);

impl<S> RangeBitset<S> {
    /// Given an array index, returns an iterator that walks to the root
    ///
    /// Note that the returned iterator does not terminate, because there could
    /// always be a larger tree above us at every point.
    fn up(mut i: usize) -> impl Iterator<Item = usize> {
        let mut log2_depth = i.trailing_ones();
        std::iter::from_fn(move || {
            let out = i;
            if i & (1 << (log2_depth + 1)) == 0 {
                i += 1 << log2_depth;
            } else {
                i -= 1 << log2_depth;
            }
            log2_depth += 1;
            Some(out)
        })
    }

    /// Given an item index range, returns an iterator that covers it exactly
    ///
    /// The iterator is specified in terms of array indexes that cover the
    /// provided range, using accumulator nodes to take large steps when
    /// possible.
    fn range_iter<R>(r: R) -> impl Iterator<Item = usize>
    where
        R: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound;

        // Starting point as an **item** index (not an array index!)
        let mut start = match r.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
        };
        // Number of items remaining
        let mut rem = match r.end_bound() {
            Bound::Unbounded => usize::MAX,
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
        }
        .saturating_sub(start);

        std::iter::from_fn(move || {
            if rem == 0 {
                return None;
            }
            // Size of the group for which `start` is the left-most item
            //
            // A size of 0 means that `start` is the right-most item in its
            // tree; a size of 1 means that we can step over a 2-element (+ 1
            // accumulator) tree; a size of 2 means that we can step over a
            // 4-element (+ 3 accumulator) tree, etc
            let log2_group_size = start.trailing_zeros();

            // This is the group size implied by our remaining item count
            //
            // For example, if we have 1 item remaining, this is 0;
            // if we have 2 or 3 items remaining, this is 1, if we have 4-7
            // items remaining, this is 2, etc.
            let log2_rem_size = std::mem::size_of_val(&rem) as u32 * 8
                - rem.leading_zeros()
                - 1;

            // How many items do we cover in this group?
            let step = 1 << log2_group_size.min(log2_rem_size);

            // Convert the output to an array index
            let out = start * 2 + step - 1;
            start += step;
            rem -= step;

            Some(out)
        })
    }
}
impl<S: Set + Copy + Clone> RangeBitset<S> {
    /// Builds a new empty data structure
    pub fn new() -> Self {
        Self(vec![])
    }

    /// Pushes the given value to the list
    pub fn push(&mut self, value: S) {
        let i = self.0.len(); // index of newly-pushed item

        // Push the actual node data to the even position
        self.0.push(Node {
            value,
            accum: S::zero(),
        });

        // Push an empty accumulator node to the odd position
        self.0.push(Node {
            value: S::zero(),
            accum: S::zero(),
        });

        // Accumulate values up the tree
        let mut value = value;
        for i in Self::up(i) {
            if let Some(n) = self.0.get_mut(i) {
                n.accum.merge(&value);
                value = n.accum;
            } else {
                break;
            }
        }
    }

    /// Returns the number of items stored in this data structure
    pub fn len(&self) -> usize {
        self.0.len() / 2
    }

    /// Computes a merged result across the given range
    pub fn range_query<R>(&self, r: R) -> S
    where
        R: std::ops::RangeBounds<usize>,
    {
        let mut out = S::zero();
        for i in Self::range_iter(r) {
            out.merge(&self.0[i].accum);
            for j in Self::up(i) {
                if let Some(v) = self.0.get(j) {
                    out.merge(&v.value);
                } else {
                    break;
                }
            }
        }

        out
    }

    /// Inserts the given value across the provided range
    pub fn update_range<R>(&mut self, r: R, new_value: S)
    where
        R: std::ops::RangeBounds<usize>,
    {
        for i in Self::range_iter(r) {
            self.0[i].value.merge(&new_value);
            for j in Self::up(i) {
                if let Some(v) = self.0.get_mut(j) {
                    v.accum.merge(&new_value);
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    impl Set for u16 {
        fn zero() -> u16 {
            0
        }
        fn merge(&mut self, v: &u16) {
            *self |= v;
        }
    }

    fn vec_and_range(
    ) -> impl Strategy<Value = (Vec<u16>, std::ops::Range<usize>)> {
        prop::collection::vec(0..u16::MAX, 1..100)
            .prop_flat_map(|vec| {
                let len = vec.len();
                (Just(vec), 0..len)
            })
            .prop_flat_map(|(vec, start)| {
                let len = vec.len();
                (Just(vec), Just(start), start..len)
            })
            .prop_flat_map(|(vec, start, end)| (Just(vec), Just(start..end)))
    }

    /// Returns a strategy for checking a modify operation
    ///
    /// The strategy is encoded in a single tuple as
    /// `(initial values, write range, value to write, read range)`
    fn modify_strategy() -> impl Strategy<
        Value = (
            Vec<u16>,
            u16,
            std::ops::Range<usize>,
            std::ops::Range<usize>,
        ),
    > {
        prop::collection::vec(0..u16::MAX, 1..1000)
            .prop_flat_map(|vec| {
                let len = vec.len();
                (Just(vec), 0..u16::MAX, 0..len, 0..len)
            })
            .prop_flat_map(|(vec, v, s1, s2)| {
                let len = vec.len();
                (Just(vec), Just(v), Just(s1), Just(s2), s1..len, s2..len)
            })
            .prop_flat_map(|(vec, v, s1, s2, e1, e2)| {
                (Just(vec), Just(v), Just(s1..e1), Just(s2..e2))
            })
    }

    fn check_bitset_construction(data: Vec<u16>, r: std::ops::Range<usize>) {
        let mut a: RangeBitset<u16> = RangeBitset::new();
        for v in &data {
            a.push(*v);
        }
        let out = a.range_query(r.clone());
        assert_eq!(out, data[r].iter().fold(0, |a, b| a | b));
    }

    fn check_bitset_modify(
        mut data: Vec<u16>,
        new_value: u16,
        write_range: std::ops::Range<usize>,
        read_range: std::ops::Range<usize>,
    ) {
        let mut a: RangeBitset<u16> = RangeBitset::new();
        for v in &data {
            a.push(*v);
        }
        a.update_range(write_range.clone(), new_value);
        for i in write_range {
            data[i] |= new_value;
        }

        let out = a.range_query(read_range.clone());
        assert_eq!(out, data[read_range].iter().fold(0, |a, b| a | b));
    }

    proptest! {
        #[test]
        fn test_bitset_construction((vec, r) in vec_and_range()) {
            check_bitset_construction(vec, r);
        }
        #[test]
        fn test_bitset_modify(
            (vec, new_value, write_range, read_range) in modify_strategy()
        ) {
            check_bitset_modify(vec, new_value, write_range, read_range);
        }
    }

    #[test]
    fn test_tree_up() {
        let mut iter = RangeBitset::<u16>::up(0);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeBitset::<u16>::up(2);
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeBitset::<u16>::up(5);
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeBitset::<u16>::up(10);
        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), Some(11));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeBitset::<u16>::up(9);
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), Some(11));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));
    }

    #[test]
    fn test_range_iter() {
        assert_eq!(
            RangeBitset::<u16>::range_iter(1..=2).collect::<Vec<_>>(),
            vec![2, 4],
        );
        assert_eq!(
            RangeBitset::<u16>::range_iter(0..=1).collect::<Vec<_>>(),
            vec![1],
        );
        assert_eq!(
            RangeBitset::<u16>::range_iter(0..=2).collect::<Vec<_>>(),
            vec![1, 4],
        );
        assert_eq!(
            RangeBitset::<u16>::range_iter(1..6).collect::<Vec<_>>(),
            vec![2, 5, 9],
        );
    }
}
