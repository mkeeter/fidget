/// Trait for a bounded meet-semilattice
pub trait Semilattice {
    /// Returns a new identity value
    ///
    /// The identity does not modify other values when combined with [`meet`]
    fn identity() -> Self;

    /// Combines two values
    ///
    /// This operator must be associative, commutative, and idempotent
    fn meet(&mut self, other: &Self);
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
//     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14     array index
//     0       1       2       3       4       5       6       7     item index
//
//     0   1   0   1   0   1   0   1   0   1   0   1   0   1   0     LSB
//     0   0   1   1   0   0   1   1   0   0   1   1   0   0   1
//     0   0   0   0   1   1   1   1   0   0   0   0   1   1   1
//     0   0   0   0   0   0   0   0   1   1   1   1   1   1   1

#[derive(Clone)]
struct Node<S> {
    /// Bottom-up values, populated by pushing individual items
    accum: S,

    /// Top-down values, populated by updating ranges
    value: S,
}

/// A type with efficient range queries and updates
///
/// This data structure stores a type that implements [`Semilattice`], which is
/// less exotic that it sounds: one example would be integers combined with
/// the `max` operator.
///
/// It supports efficient aggregate range queries, e.g. finding the `max` across
/// a range of values; in addition, it supports range **updates**.
///
/// Under the hood, this is implemented as a modified implicit in-order forest;
/// see [this blog post](https://www.mattkeeter.com/blog/2023-07-03-iforest/)
/// for details and further citations.
#[derive(Clone, Default)]
pub struct RangeData<S>(Vec<Node<S>>);

impl<S> RangeData<S> {
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

impl<S: Semilattice + Copy + Clone> RangeData<S> {
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
            accum: S::identity(),
        });

        // Push an empty accumulator node to the odd position
        self.0.push(Node {
            value: S::identity(),
            accum: S::identity(),
        });

        // Accumulate values up the tree
        let mut value = value;
        for i in Self::up(i) {
            if let Some(n) = self.0.get_mut(i) {
                n.accum.meet(&value);
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

    /// Computes an aggregate result across the given range
    pub fn range_query<R>(&self, r: R) -> S
    where
        R: std::ops::RangeBounds<usize>,
    {
        let mut out = S::identity();
        for i in Self::range_iter(r) {
            out.meet(&self.0[i].accum);
            for j in Self::up(i) {
                if let Some(v) = self.0.get(j) {
                    out.meet(&v.value);
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
            self.0[i].value.meet(&new_value);
            for j in Self::up(i) {
                if let Some(v) = self.0.get_mut(j) {
                    v.accum.meet(&new_value);
                } else {
                    break;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Bitset representing active registers
///
/// This is 256 bits wide, since we use a `u8` to store a register index
#[derive(Copy, Clone, Default)]
pub struct RegisterSet([u32; 8]);

impl RegisterSet {
    /// Returns a new empty register set
    pub fn new() -> Self {
        Self::identity()
    }

    /// Marks the given register as present
    ///
    /// # Panics
    /// `i` must be within the valid range (i.e. `i < 256`)
    pub fn insert(&mut self, i: usize) {
        self.0[i / 32] |= 1 << (i % 32);
    }

    /// Returns the lowest available register, or `None`
    pub fn available(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find_map(|(i, r)| match r.trailing_ones() {
                32 => None,
                n => Some(i as usize * 32 + n as usize),
            })
    }
}

impl Semilattice for RegisterSet {
    fn identity() -> Self {
        Self([0; 8])
    }
    fn meet(&mut self, other: &RegisterSet) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= *b;
        }
    }
}

/// Active registers across time, with efficient range queries and updates
pub type ActiveRegisterRange = RangeData<RegisterSet>;

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    impl Semilattice for u16 {
        fn identity() -> u16 {
            0
        }
        fn meet(&mut self, v: &u16) {
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
        let mut a: RangeData<u16> = RangeData::new();
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
        let mut a: RangeData<u16> = RangeData::new();
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
        let mut iter = RangeData::<u16>::up(0);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeData::<u16>::up(2);
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeData::<u16>::up(5);
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeData::<u16>::up(10);
        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), Some(11));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));

        let mut iter = RangeData::<u16>::up(9);
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), Some(11));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(15));
    }

    #[test]
    fn test_range_iter() {
        assert_eq!(
            RangeData::<u16>::range_iter(1..=2).collect::<Vec<_>>(),
            vec![2, 4],
        );
        assert_eq!(
            RangeData::<u16>::range_iter(0..=1).collect::<Vec<_>>(),
            vec![1],
        );
        assert_eq!(
            RangeData::<u16>::range_iter(0..=2).collect::<Vec<_>>(),
            vec![1, 4],
        );
        assert_eq!(
            RangeData::<u16>::range_iter(1..6).collect::<Vec<_>>(),
            vec![2, 5, 9],
        );
    }

    #[test]
    fn test_registerset() {
        let mut r = RegisterSet::new();
        assert_eq!(r.available(), Some(0));
        r.insert(1);
        assert_eq!(r.available(), Some(0));
        r.insert(0);
        assert_eq!(r.available(), Some(2));
        r.insert(2);
        assert_eq!(r.available(), Some(3));
        for i in 0..256 {
            r.insert(i);
        }
        assert_eq!(r.available(), None);
    }
}
