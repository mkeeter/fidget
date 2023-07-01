//! Choice bitsets

/// A choice index is a base offset into the bitset along with a bit offset
///
/// The base offset is needed because choices need to know where a particular
/// choice set **starts**; this allows us to look up whether a value has already
/// been written when doing an in-place `min` or `max` operation.
#[derive(Copy, Clone, Debug)]
pub struct ChoiceIndex {
    pub index: u16,
    pub bit: u16,
}

impl ChoiceIndex {
    pub fn new(index: usize, bit: usize) -> Self {
        Self {
            index: index.try_into().unwrap(),
            bit: bit.try_into().unwrap(),
        }
    }
    pub fn start(&self) -> usize {
        self.index as usize
    }
    pub fn end(&self) -> usize {
        (self.index + self.bit / 8) as usize
    }
}

/// Stores a group of choice sets, using a `[u8]`-shaped type for storage
///
/// The inner type could be either owned or borrowed (i.e. a `&mut [u8]` or a
/// `Vec<u8>`), depending on use case.
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
pub struct Choices(Vec<u8>);

impl Choices {
    /// Builds a new empty choice set
    pub fn new() -> Self {
        Self::default()
    }

    /// Resizes and fills with the given value
    pub fn resize_and_fill(&mut self, size: usize, t: u8) {
        self.0.resize(size, t);
        self.0.fill(t);
    }

    /// Returns an immutable view into the choice data array
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// Returns the inner data array
    pub fn take(self) -> Vec<u8> {
        self.0
    }

    /// Checks whether the given choice has already received a value
    pub fn has_value(&self, c: ChoiceIndex) -> bool {
        (c.start()..=c.end()).any(|i| self.0[i] != 0)
    }

    /// Sets the given index
    pub fn set(&mut self, c: ChoiceIndex) {
        self.0[c.end()] |= 1 << (c.bit % 8);
    }

    /// Clears all previous bits and sets the given index
    pub fn set_exclusive(&mut self, c: ChoiceIndex) {
        for i in c.start()..=c.end() {
            self.0[i] = 0;
        }
        self.set(c);
    }
}
