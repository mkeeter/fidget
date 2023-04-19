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
        let _bad_axis_const = [0u8; 1][(i.count_ones() != 1) as usize];
        let _bad_axis = [0u8; 1][(i.trailing_zeros() >= 3) as usize];
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

impl std::ops::BitAnd<Corner> for Axis {
    type Output = bool;
    fn bitand(self, rhs: Corner) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl std::ops::BitOr<Axis> for Axis {
    type Output = Corner;
    fn bitor(self, rhs: Axis) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl std::ops::BitOr<Corner> for Axis {
    type Output = Corner;
    fn bitor(self, rhs: Corner) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl From<Axis> for Corner {
    fn from(a: Axis) -> Self {
        Corner::new(a.0)
    }
}

/// Strongly-typed cell corner, in the 0-8 range
#[derive(Copy, Clone, Debug)]
pub struct Corner(u8);

impl Corner {
    /// Builds a new corner
    ///
    /// # Panics
    /// If `i >= 8`, which is not a valid cube corner index
    pub const fn new(i: u8) -> Self {
        assert!(i < 8);
        Self(i)
    }
    /// Returns the value of this corner as an index
    pub fn index(self) -> usize {
        self.0 as usize
    }
    /// Iterates over all 8 corners
    pub fn iter() -> impl Iterator<Item = Corner> {
        (0..8).map(Corner)
    }
}

impl std::ops::BitAnd<Axis> for Corner {
    type Output = bool;
    fn bitand(self, rhs: Axis) -> bool {
        (self.0 & rhs.0) != 0
    }
}

impl std::ops::BitOr<Corner> for Corner {
    type Output = Corner;
    fn bitor(self, rhs: Corner) -> Corner {
        Corner(self.0 | rhs.0)
    }
}

impl std::ops::BitOr<Axis> for Corner {
    type Output = Corner;
    fn bitor(self, rhs: Axis) -> Corner {
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
    start: Corner,
    /// Ending corner
    end: Corner,
}

impl DirectedEdge {
    /// Builds a new directed edge
    ///
    /// # Panics
    /// If the start and end aren't the same
    ///
    /// ```compile_fail
    /// # use fidget::mesh::types::{Corner, DirectedEdge};
    /// const START: Corner = Corner::new(0);
    /// const E: DirectedEdge = DirectedEdge::new(START, START);
    /// ```
    pub const fn new(start: Corner, end: Corner) -> Self {
        let _corners_must_not_match = [0; 1][(start.0 == end.0) as usize];
        Self { start, end }
    }
    /// Returns the start corner
    pub fn start(self) -> Corner {
        self.start
    }
    /// Returns the end corner
    pub fn end(self) -> Corner {
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
        let _bad_index_const = [0; 12][i as usize];
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
    pub fn corners(&self) -> (Corner, Corner) {
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

/// Face mask, as an 4-bit value representing set corners
///
/// This value is bound to a particular [`Frame`] and is meaningless in
/// isolation.  Within that frame `(t, u, v)`, the corners are encoded as
///
///   Bit | Corner
///   ----|--------
///    0  | 0
///    1  | u
///    2  | v
///    3  | u | v
///
/// (i.e. `u` is bit 0 of the mask and `v` is bit 1 of the mask)
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FaceMask(pub u8);

impl FaceMask {
    pub const fn new(i: u8) -> Self {
        let _bad_face_mask = [0u8; 1][(i >= 16) as usize];
        Self(i)
    }
}

/// Edge mask, as an 2-bit value representing set corners
///
/// This value is bound to a particular [`Frame`] and is meaningless in
/// isolation.  Within that frame `(t, u, v)`, the corners are encoded as
///
///   Bit | Corner
///   ----|--------
///    0  | 0
///    1  | t
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EdgeMask(pub u8);

impl EdgeMask {
    pub const fn new(i: u8) -> Self {
        let _bad_face_mask = [0u8; 1][(i >= 4) as usize];
        Self(i)
    }
}

/// Face index, as a value in the range 0-5
///
/// Encoded as `axis * 2 + sign`, where `axis` is a value in the range 0-2
/// and sign is `+1` for the face in the `+axis` direction.
#[derive(Copy, Clone, Debug)]
pub struct Face(u8);

impl Face {
    pub const fn new(i: u8) -> Self {
        let _bad_face = [0u8; 1][(i >= 6) as usize];
        Self(i)
    }

    /// Extracts the main axis of this face
    pub const fn axis(self) -> Axis {
        Axis::new(1 << (self.0 / 2))
    }

    /// Returns the sign of this face, which is a boolean
    ///
    /// When true, the face is on the `+axis` side of the cell
    pub const fn sign(self) -> bool {
        self.0 % 2 != 0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_face_axis() {
        assert_eq!(Face::new(0).axis(), X);
        assert_eq!(Face::new(1).axis(), X);
        assert_eq!(Face::new(2).axis(), Y);
        assert_eq!(Face::new(3).axis(), Y);
        assert_eq!(Face::new(4).axis(), Z);
        assert_eq!(Face::new(5).axis(), Z);
    }
}
