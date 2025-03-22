//! Strongly-typed indexes of various flavors
//!
//! Users are unlikely to actually use types within this crate; it's public
//! because there are certain properties which need to be tested within a
//! `compile_fail` doctest.

/// A single axis, represented as a `u8` with one bit set
///
/// An `Axis<2>` has bit 0 or 1 set; an `Axis<3>` as bit 0, 1, or 2 set.
///
/// This invariant is enforced at construction
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Axis<const D: usize>(u8);

impl<const D: usize> Axis<D> {
    /// Builds a new axis
    ///
    /// ```
    /// # use fidget::mesh::types::Axis;
    /// const X: Axis<3> = Axis::new(1);
    /// const Y: Axis<3> = Axis::new(2);
    /// const Z: Axis<3> = Axis::new(4);
    /// ```
    ///
    /// # Panics
    /// If the input does not have exactly 1 set bit
    ///
    /// ```compile_fail
    /// # use fidget::mesh::types::Axis;
    /// const A: Axis = Axis::new(0b11);
    /// ```
    ///
    /// If the input has a bit set that's outside the valid range
    /// ```compile_fail
    /// # use fidget::mesh::types::Axis;
    /// const A: Axis<3> = Axis::new(0b1000);
    /// ```
    pub const fn new(i: u8) -> Self {
        assert!(i.count_ones() == 1);
        assert!(i.trailing_zeros() < D as u32);
        Self(i)
    }

    /// Converts from a bitmask to an index
    pub fn index(self) -> usize {
        self.0.trailing_zeros() as usize
    }

    /// Returns an array of valid axes at this dimension
    pub const fn array() -> [Self; D] {
        // NOTE: this breaks invariants, but we're going to overwrite them
        let mut out = [Axis(0); D];
        let mut i = 0;
        loop {
            if i == D {
                break;
            }
            out[i] = Axis::new(1 << i);
            i += 1;
        }
        out
    }
}

impl Axis<3> {
    /// Cycles through X-Y-Z axes, returning the next one
    pub const fn next(self) -> Self {
        let u = self.0 << 1;
        if u > Z.0 {
            X
        } else {
            Axis(u)
        }
    }
}

/// The X axis, i.e. `[1, 0, 0]`
pub const X: Axis<3> = Axis(1);
/// The Y axis, i.e. `[0, 1, 0]`
pub const Y: Axis<3> = Axis(2);
/// The Z axis, i.e. `[0, 0, 1]`
pub const Z: Axis<3> = Axis(4);

impl<const D: usize> std::ops::Mul<bool> for Axis<D> {
    type Output = Self;
    fn mul(self, rhs: bool) -> Self {
        if rhs {
            self
        } else {
            Axis(0)
        }
    }
}

impl<const D: usize> std::ops::BitAnd<Corner<D>> for Axis<D> {
    type Output = bool;
    fn bitand(self, rhs: Corner<D>) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl<const D: usize> std::ops::BitOr<Axis<D>> for Axis<D> {
    type Output = Corner<D>;
    fn bitor(self, rhs: Axis<D>) -> Self::Output {
        Corner::new(self.0 | rhs.0)
    }
}

impl<const D: usize> std::ops::BitOr<Corner<D>> for Axis<D> {
    type Output = Corner<D>;
    fn bitor(self, rhs: Corner<D>) -> Self::Output {
        Corner::new(self.0 | rhs.0)
    }
}

impl<const D: usize> From<Axis<D>> for Corner<D> {
    fn from(a: Axis<D>) -> Self {
        Corner::new(a.0)
    }
}

/// Strongly-typed cell corner, in the `[0, 2**D)` range
#[derive(Copy, Clone, Debug)]
pub struct Corner<const D: usize>(u8);

impl<const D: usize> Corner<D> {
    /// Builds a new corner
    ///
    /// # Panics
    /// If `i >= 8`, which is not a valid corner index
    pub const fn new(i: u8) -> Self {
        assert!(i < (1 << D));
        Self(i)
    }
    /// Returns the value of this corner as an index
    pub fn index(self) -> usize {
        self.0 as usize
    }
    /// Returns the value of this corner as a `u8`
    pub fn get(self) -> u8 {
        self.0
    }
    /// Iterates over all 8 corners
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..(1 << D)).map(Corner)
    }
}

impl<const D: usize> std::ops::BitAnd<Axis<D>> for Corner<D> {
    type Output = bool;
    fn bitand(self, rhs: Axis<D>) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl<const D: usize> std::ops::BitOr<Corner<D>> for Corner<D> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Corner::new(self.0 | rhs.0)
    }
}

impl<const D: usize> std::ops::BitOr<Axis<D>> for Corner<D> {
    type Output = Self;
    fn bitor(self, rhs: Axis<D>) -> Self {
        Corner::new(self.0 | rhs.0)
    }
}

/// A directed edge within an octree cell
///
/// This data structure enforces the invariant that the start and end must be
/// different (checked during construction).
#[derive(Copy, Clone, Debug)]
pub struct DirectedEdge {
    /// Starting corner
    start: Corner<3>,
    /// Ending corner
    end: Corner<3>,
}

impl DirectedEdge {
    /// Builds a new directed edge within a cube
    ///
    /// # Panics
    /// If the start and end aren't the same
    ///
    /// ```compile_fail
    /// # use fidget::mesh::types::{Corner, DirectedEdge};
    /// const START: Corner = Corner::new(0);
    /// const E: DirectedEdge = DirectedEdge::new(START, START);
    /// ```
    ///
    /// If the start and end point are different on more than one axis (e.g.
    /// there are no diagonal edges in a cube):
    ///
    /// ```compile_fail
    /// # use fidget::mesh::types::{Corner, DirectedEdge};
    /// const START: Corner = Corner::new(0);
    /// const END: Corner = Corner::new(0b111);
    /// const E: DirectedEdge = DirectedEdge::new(START, END);
    /// ```
    pub const fn new(start: Corner<3>, end: Corner<3>) -> Self {
        assert!(start.0 != end.0);
        assert!((start.0 ^ end.0).count_ones() == 1);
        Self { start, end }
    }
    /// Returns the start corner
    pub fn start(self) -> Corner<3> {
        self.start
    }
    /// Returns the end corner
    pub fn end(self) -> Corner<3> {
        self.end
    }
    pub fn to_undirected(self) -> Edge {
        let t = Axis(self.start.0 ^ self.end.0);
        let u = t.next();
        let v = u.next();

        #[allow(clippy::bool_to_int_with_if)]
        Edge::new(
            (t.0.trailing_zeros() as u8) * 4
                + if self.start & v { 2 } else { 0 }
                + if self.start & u { 1 } else { 0 },
        )
    }
}

/// An undirected edge within an octree cell
///
/// With `(t, u, v)` as a right-handed coordinate system and `t` being the
/// varying axis of the edge, this is packed as `4 * t + 2 * v + 1 * u`
/// (where `t`, `u`, and `v` are values in the range 0-2 representing an axis)
#[derive(Copy, Clone, Debug)]
pub struct Edge(u8);

impl Edge {
    /// Builds a new edge
    ///
    /// # Panics
    /// If `i >= 12`, since that's an invalid edge
    pub const fn new(i: u8) -> Self {
        assert!(i < 12);
        Self(i)
    }
    /// Converts from an edge to an index
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// Returns a `(start, end)` tuple for the given edge
    ///
    /// In the `t, u, v` coordinate system, the start always the `t` bit clear
    /// and the end always has the `t` bit set; the `u` and `v` bits are the
    /// same at both start and end.
    pub fn corners(&self) -> (Corner<3>, Corner<3>) {
        use super::frame::{Frame, XYZ, YZX, ZXY};
        let (t, u, v) = match self.0 / 4 {
            0 => XYZ::frame(),
            1 => YZX::frame(),
            2 => ZXY::frame(),
            _ => unreachable!("invalid edge index"),
        };

        let u = u * ((self.0 % 4) % 2 != 0);
        let v = v * ((self.0 % 4) / 2 != 0);

        (u | v, t | u | v)
    }
}

/// Represents the relative offset of a vertex within `Octree::verts`
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Offset(pub u8);

/// Information about how to mesh an edge that contains a sign change
#[derive(Copy, Clone, Debug)]
pub struct Intersection {
    /// Data offset of the vertex located within the cell
    pub vert: Offset,
    /// Data offset of the vertex located on the edge
    pub edge: Offset,
}

/// Bitmask of which corners in a cell are inside the shape
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CellMask<const D: usize>(u8);

impl<const D: usize> CellMask<D> {
    const MASK: u8 = ((1u16 << (1 << D)) - 1) as u8;

    /// Builds a new `CellMask`
    ///
    /// # Panics
    /// If invalid bits are set in the mask.  This can't happen for
    /// `CellMask<3>`, but is possible for `CellMask<2>` if any of the upper 4
    /// bits are set.
    ///
    /// ```should_panic
    /// # use fidget::mesh::types::{CellMask};
    /// let m = CellMask::<2>::new(0b11111111);
    /// ```
    pub const fn new(i: u8) -> Self {
        if i & !Self::MASK != 0 {
            panic!();
        }
        Self(i)
    }

    /// Returns the bitmask as an index
    ///
    /// The index has the same value as the bitmask, but is cast to a `usize`
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    pub fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }
}

impl<const N: usize> std::ops::BitAnd<Corner<N>> for CellMask<N> {
    type Output = bool;
    fn bitand(self, c: Corner<N>) -> bool {
        (self.0 & (1 << c.index())) != 0
    }
}
