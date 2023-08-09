//! Choice bitsets

/// A choice index is a base offset into the bitset along with a bit offset
///
/// The base offset is needed because choices need to know where a particular
/// choice set **starts**; this allows us to look up whether a value has already
/// been written when doing an in-place `min` or `max` operation.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ChoiceIndex {
    pub index: u16,
    pub bit: u16,
}

/// A masked lookup for use in the [`Choices`] array
///
/// This may span multiple choice indices, but that's fine.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ChoiceMask {
    pub index: usize,
    pub mask: u64,
}

impl ChoiceIndex {
    pub fn new(index: usize, bit: usize) -> Self {
        Self {
            index: index.try_into().unwrap(),
            bit: bit.try_into().unwrap(),
        }
    }
    pub fn start(&self) -> u16 {
        self.index
    }
    pub fn end(&self) -> u16 {
        (self.index + self.bit / 8) as u16
    }
    pub fn mask(&self) -> u8 {
        1 << (self.bit % 8)
    }
}

/// Stores a group of choice sets, using a `[u64]`-shaped type for storage
///
/// The inner type could be either owned or borrowed (i.e. a `&mut [u64]` or a
/// `Vec<u64>`), depending on use case.
///
/// Each individual choice set is represented as a range of bits within the
/// data array, aligned to start at a `u8` boundary.  (For example, a 2-bit
/// choice set would still take up an entire `u8`)
///
/// An individual choice set within the array has the same `ChoiceIndex::index`
/// value; individual choices within that set vary in their `ChoiceIndex::bit`.
/// Operations upon a choice set with the same `index` are assumed to run in
/// ascending `bit` order.
#[derive(Default)]
pub struct Choices {
    data: Vec<u64>,
}

impl Choices {
    /// Builds a new empty choice set
    pub fn new() -> Self {
        Self::default()
    }

    /// Resizes and fills with the given value
    pub fn resize_and_zero(&mut self, size: usize) {
        self.data.resize(size, 0);
        self.data.fill(0);
    }

    /// Returns an mutable view into the choice data array
    pub fn as_mut(&mut self) -> &mut [u64] {
        self.finalize();
        self.data.as_mut()
    }

    /// Returns the inner data array
    pub fn take(mut self) -> Vec<u64> {
        self.finalize();
        self.data
    }

    /// Checks whether the given choice has already received a value
    pub fn has_value(&self, c: ChoiceIndex) -> bool {
        assert_eq!(c.bit % 2, 0);
        self[c.index] & 3 == 3
    }

    pub fn is_set(&self, c: ChoiceIndex) -> bool {
        (self[c.end()] & c.mask()) != 0
    }

    pub fn set_has_value(&mut self, c: ChoiceIndex) {
        self[c.start()] = 3;
    }

    /// Sets the given index
    pub fn set(&mut self, c: ChoiceIndex) {
        debug_assert_eq!(c.bit % 2, 0);
        self[c.end()] |= c.mask();
        self[c.start()] |= 3;
    }

    /// Clears all previous bits and sets the given index
    pub fn set_exclusive(&mut self, c: ChoiceIndex) {
        debug_assert_eq!(c.bit % 2, 0);
        self[c.start()] |= 3;
        // If this exclusivity is contained within a single byte, then we don't
        // need to post-process it later.
        if c.start() == c.end() {
            debug_assert!(c.bit < 8);
            // Clear all lower bits, then set the inclusive bit
            self[c.start()] = c.mask() | 3;
        } else {
            // Clear all lower bits, then set the *exclusive* bit
            // This signals that `finalize` must iterate backwards to the start
            self[c.end()] = c.mask() << 1;
            self[c.start()] |= 3;
        }
    }

    fn finalize(&mut self) {
        let mut i = self.data.len();
        while i > 0 {
            i -= 1;
            let d = self.data[i];
            let has_initial = d & 0x3333_3333_3333_3333;

            // Each byte has its LSB set iff it's an initial byte
            let initial = has_initial & has_initial >> 1;

            // A byte contains an exclusive set if an odd bit is set in that
            // byte, and that odd bit isn't part of an initial-byte marker
            let has_exclusive = (d & 0xAAAA_AAAA_AAAA_AAAA) & !(initial << 1);

            if has_exclusive != 0 {
                let leading_zeros = has_exclusive.leading_zeros();
                let highest_exclusive_bit = 64 - 1 - leading_zeros;

                // Shift to byte-wise iteration for simplicity here
                let mut j = (i * 8 + highest_exclusive_bit as usize / 8)
                    .try_into()
                    .unwrap();
                let highest_exclusive_bit = highest_exclusive_bit % 8;

                // Move the exclusive bit to the inclusive position
                self[j] &= !(1 << highest_exclusive_bit);
                self[j] &= !((1 << highest_exclusive_bit) - 1);
                self[j] |= 1 << (highest_exclusive_bit - 1);

                while j > 0 {
                    j -= 1;
                    if self[j] & 3 == 3 {
                        self[j] = 3;
                        break;
                    } else {
                        self[j] = 0;
                    }
                }
                // There may be multiple exclusive bits in the same u64, so
                // restore i to its previous value.
                i += 1;
            }
        }
    }
}

impl std::ops::Index<u16> for Choices {
    type Output = u8;
    fn index(&self, i: u16) -> &u8 {
        // SAFETY:
        // Casting a `&[u64]` to a `&[u8]` is always allowed
        //
        // Note that this code could use `std::slice::align_as`, but in
        // practice, that doesn't get inlined out and we see a large amount of
        // `core::ptr::align_offset` in our trace.
        let s = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *mut u8,
                self.data.len() * 8,
            )
        };
        &s[i as usize]
    }
}

impl std::ops::IndexMut<u16> for Choices {
    fn index_mut(&mut self, i: u16) -> &mut u8 {
        // SAFETY:
        // Casting a `&[u64]` to a `&[u8]` is always allowed
        let s = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                self.data.len() * 8,
            )
        };
        &mut s[i as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_choices() {
        let mut c = Choices::new();
        c.resize_and_zero(1);
        c.set(ChoiceIndex::new(0, 2));
        assert_eq!(c.as_mut(), &mut [7]);
        c.set(ChoiceIndex::new(0, 4));
        assert_eq!(c.as_mut(), &mut [7 | 0b010000]);
        c.set(ChoiceIndex::new(0, 6));
        assert_eq!(c.as_mut(), &mut [7 | 0b01010000]);
        c.set_exclusive(ChoiceIndex::new(0, 8));
        assert_eq!(c.as_mut(), &mut [3 | 0b0100000000]);
    }
}
