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
        self.data.as_mut()
    }

    /// Returns the inner data array
    pub fn take(mut self) -> Vec<u64> {
        self.data
    }

    /// Checks whether the given choice has already received a value
    pub fn has_value(&self, c: ChoiceIndex) -> bool {
        self[c.index] & 1 == 1
    }

    pub fn is_set(&self, c: ChoiceIndex) -> bool {
        (self[c.end()] & c.mask()) != 0
    }

    pub fn set_has_value(&mut self, c: ChoiceIndex) {
        self[c.start()] = 1;
    }

    /// Sets the given index
    pub fn set(&mut self, c: ChoiceIndex) {
        self[c.start()] |= 1;
        self[c.end()] |= c.mask();
    }

    /// Clears all previous bits and sets the given index
    pub fn set_exclusive(&mut self, c: ChoiceIndex) {
        // If this exclusivity is contained within a single byte, then we don't
        // need to post-process it later.
        if c.start() == c.end() {
            debug_assert!(c.bit < 8);
            // Clear all lower bits, then set the inclusive bit
            self[c.start()] = c.mask() | 1;
        } else {
            self[c.start()] = 1;
            for i in c.start() + 1..c.end() {
                self[i] = 0;
            }
            self[c.end()] = c.mask();
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
        assert_eq!(c.as_mut(), &mut [5]);
        c.set(ChoiceIndex::new(0, 4));
        assert_eq!(c.as_mut(), &mut [21]);
        c.set(ChoiceIndex::new(0, 6));
        assert_eq!(c.as_mut(), &mut [85]);
        c.set_exclusive(ChoiceIndex::new(0, 8));
        assert_eq!(c.as_mut(), &mut [257]);
    }
}
