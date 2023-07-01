#[derive(Copy, Clone, Default)]
struct RegisterSet([u32; 8]);

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

/// Set of active registers with efficient range queries
///
/// Under the hood, this is implemented as an implicit in-order forest; see
/// [the original post](https://thume.ca/2021/03/14/iforests/) and
/// [https://github.com/havelessbemore/dastal/blob/main/src/segmentTree/inOrderSegmentTree.md](this excellent writeup)
/// for details.
struct ActiveRegisters<S>(Vec<Node<S>>);

impl<S: Set + Copy + Clone> ActiveRegisters<S> {
    pub fn new() -> Self {
        Self(vec![])
    }
    pub fn push(&mut self, d: S) {
        let mut i = self.0.len(); // index of newly-pushed item
        let bits = i / 2;

        // Push the actual node data to the even position
        self.0.push(Node {
            value: S::zero(),
            accum: d,
        });

        // Push an empty accumulator node to the odd position
        self.0.push(Node {
            value: S::zero(),
            accum: S::zero(),
        });

        // Merge left-ward until we run out of bits
        let mut scale = 1;
        while bits & scale != 0 {
            let mut accum = self.0[i].accum;
            accum.merge(&self.0[i - scale * 2].accum);
            self.0[i - scale] = Node {
                accum,
                value: S::zero(),
            };
            i -= scale;
            scale <<= 2;
        }
    }

    pub fn len(&self) -> usize {
        self.0.len() / 2
    }

    pub fn range_query<R>(&self, r: R) -> S
    where
        R: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound;
        // Inclusive start, exclusive end
        let mut start = match r.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
        };
        let mut rem = match r.end_bound() {
            Bound::Unbounded => self.len(),
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
        }
        .saturating_sub(start);

        let mut accum = S::zero();
        println!("looping {rem}");
        while rem != 0 {
            // Find the largest subtree that we can step over without exceeding
            // our range.  This is equivalent to the largest subtree for which
            // `start` is the left-most leaf.
            let subtree_size = start.trailing_zeros();
            let step = subtree_size.min(
                std::mem::size_of::<usize>() as u32 * 8
                    - rem.leading_zeros()
                    - 1,
            ) as usize; // TODO use a bitmask here?

            // root of a subtree
            let mut root = start * 2 + step;

            println!("root: {root}");
            println!("step: {step}");
            println!("subtree size: {subtree_size}");
            println!("z: {}", rem.leading_zeros());
            accum.merge(&self.0[root].accum);

            // walk up the subtree to the root, accumulating values
            let mut depth = root.trailing_ones();
            println!("depth: {depth}");
            while root < self.0.len() {
                accum.merge(&self.0[root].value);
                if root & (1 << (depth + 1)) == 0 {
                    root += 1 << depth;
                } else {
                    root -= 1 << depth;
                }
                depth += 1;
            }

            start += step;
            rem -= 1 << step;
        }
        accum
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
        prop::collection::vec(0..u16::MAX, 1..8)
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

    fn test_active_registers(data: Vec<u16>, r: std::ops::Range<usize>) {
        let mut a: ActiveRegisters<u16> = ActiveRegisters::new();
        for v in &data {
            a.push(*v);
        }
        let out = a.range_query(r.clone());
        assert_eq!(out, data[r].iter().fold(0, |a, b| a | b));
    }

    #[test]
    fn known_regression() {
        let d = vec![
            65347, 15708, 32604, 65450, 59352, 13374, 39410, 21793, 9702,
            48901, 23004, 26309, 27198, 17124, 13306, 6079, 19183, 47361,
            44498, 60088, 4006, 31971, 15674, 43660, 35786, 64074, 37572,
            51562, 24642, 44036, 20726, 63153, 17380, 6800, 36083, 22411,
            27209, 47949, 62623, 5227, 16509, 65378, 42226, 19805, 44364,
            15740, 46852, 49608, 19850, 35082, 46219, 27755, 49378, 50069,
            28372, 50791, 2926, 13199, 41344, 40411, 4807, 5303, 64073, 19636,
            1596, 45643, 38641, 1465, 42958, 58188, 27862, 48550, 38167, 39198,
            38615, 6522, 41162, 59990, 32567,
        ];
        let r = 4..5;
        test_active_registers(d, r);
    }

    proptest! {
        #[test]
        fn test_some_function((vec, r) in vec_and_range()) {
            test_active_registers(vec, r);
        }
    }
}
