//! Strongly-typed indexes of various flavors
//!
//! Users are unlikely to actually use types within this crate; it's public
//! because there are certain properties which need to be tested within a
//! `compile_fail` doctest.

/// A single axis, represented as a `u8` with one bit (between 0 and 3) set
///
/// These invariants are enforced at construction
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Axis(u8);

impl Axis {
    /// Builds a new axis
    ///
    /// ```
    /// # use fidget::mesh::types::Axis;
    /// const X: Axis = Axis::new(1);
    /// const Y: Axis = Axis::new(2);
    /// const Z: Axis = Axis::new(4);
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
    /// If the input has a bit set that's not in the 0-2 range
    /// ```compile_fail
    /// # use fidget::mesh::types::Axis;
    /// const A: Axis = Axis::new(0b1000);
    /// ```
    pub const fn new(i: u8) -> Self {
        assert!(i.count_ones() == 1);
        assert!(i.trailing_zeros() < 3);
        Self(i)
    }

    /// Converts from a bitmask to an index
    pub fn index(self) -> usize {
        self.0.trailing_zeros() as usize
    }

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
pub const X: Axis = Axis(1);
/// The Y axis, i.e. `[0, 1, 0]`
pub const Y: Axis = Axis(2);
/// The Z axis, i.e. `[0, 0, 1]`
pub const Z: Axis = Axis(4);

impl std::ops::Mul<bool> for Axis {
    type Output = Axis;
    fn mul(self, rhs: bool) -> Axis {
        if rhs {
            self
        } else {
            Axis(0)
        }
    }
}

impl<const D: usize> std::ops::BitAnd<Corner<D>> for Axis {
    type Output = bool;
    fn bitand(self, rhs: Corner<D>) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl std::ops::BitOr<Axis> for Axis {
    type Output = Corner<3>;
    fn bitor(self, rhs: Axis) -> Self::Output {
        Corner(self.0 | rhs.0)
    }
}

impl<const D: usize> std::ops::BitOr<Corner<D>> for Axis {
    type Output = Corner<D>;
    fn bitor(self, rhs: Corner<D>) -> Self::Output {
        Corner(self.0 | rhs.0)
    }
}

impl<const D: usize> From<Axis> for Corner<D> {
    fn from(a: Axis) -> Self {
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
    /// Iterates over all 8 corners
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..(1 << D)).map(Corner)
    }
}

impl<const D: usize> std::ops::BitAnd<Axis> for Corner<D> {
    type Output = bool;
    fn bitand(self, rhs: Axis) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl<const D: usize> std::ops::BitOr<Corner<D>> for Corner<D> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Corner(self.0 | rhs.0)
    }
}

impl<const D: usize> std::ops::BitOr<Axis> for Corner<D> {
    type Output = Self;
    fn bitor(self, rhs: Axis) -> Self {
        Corner(self.0 | rhs.0)
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
    pub fn new(i: u8) -> Self {
        assert_eq!(i & Self::MASK, i, "invalid bits set in {i}");
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
