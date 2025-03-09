//! Data types used in the octree
use crate::types::Interval;

use super::{
    gen::CELL_TO_EDGE_TO_VERT,
    types::{Axis, Corner, Edge, Intersection, X, Y, Z},
};

/// Raw cell data
///
/// Unpack to a [`Cell`] to actually use it
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct CellData(u64);

impl From<Cell> for CellData {
    fn from(c: Cell) -> Self {
        let i = match c {
            Cell::Invalid => 0,
            Cell::Empty => 1,
            Cell::Full => 2,
            Cell::Branch { index, thread } => {
                debug_assert!((index as u64) < (1 << 54));
                (0b10 << 62) | ((thread as u64) << 54) | index as u64
            }
            Cell::Leaf(Leaf { mask, index }) => {
                debug_assert!((index as u64) < (1 << 54));
                (0b11 << 62) | ((mask as u64) << 54) | index as u64
            }
        };
        CellData(i)
    }
}

impl std::fmt::Debug for CellData {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        let c: Cell = (*self).into();
        c.fmt(f)
    }
}

/// Unpacked form of [`CellData`]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Cell {
    Invalid,
    Empty,
    Full,
    Branch { index: usize, thread: u8 },
    Leaf(Leaf),
}

impl Cell {
    /// Checks whether the given corner is empty (`false`) or full (`true`)
    ///
    /// # Panics
    /// If the cell is a branch or invalid
    pub fn corner(self, c: Corner) -> bool {
        let t = 1 << c.index();
        match self {
            Cell::Leaf(Leaf { mask, .. }) => mask & t != 0,
            Cell::Empty => false,
            Cell::Full => true,
            Cell::Branch { .. } | Cell::Invalid => panic!(),
        }
    }
}

impl From<CellData> for Cell {
    fn from(c: CellData) -> Self {
        let i = c.0;
        match i {
            0 => Cell::Invalid,
            1 => Cell::Empty,
            2 => Cell::Full,
            _ => {
                let index = (i & ((1 << 54) - 1)).try_into().unwrap();
                match (i >> 62) & 0b11 {
                    0b10 => Cell::Branch {
                        thread: (i >> 54) as u8,
                        index,
                    },
                    0b11 => Cell::Leaf(Leaf {
                        mask: (i >> 54) as u8,
                        index,
                    }),
                    _ => panic!("invalid cell encoding"),
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Leaf {
    pub mask: u8, // TODO make this a stronger type, e.g. CellMask?
    pub index: usize,
}

impl Leaf {
    /// Returns the edge intersection for the given edge (if present)
    pub fn edge(&self, e: Edge) -> Option<Intersection> {
        CELL_TO_EDGE_TO_VERT[self.mask as usize][e.index()]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CellVertex {
    /// Position of this vertex
    pub pos: nalgebra::Vector3<f32>,
}

impl Default for CellVertex {
    fn default() -> Self {
        Self {
            pos: nalgebra::Vector3::new(f32::NAN, f32::NAN, f32::NAN),
        }
    }
}

impl std::ops::Index<Axis> for CellVertex {
    type Output = f32;

    fn index(&self, axis: Axis) -> &Self::Output {
        match axis {
            X => &self.pos.x,
            Y => &self.pos.y,
            Z => &self.pos.z,
            _ => panic!("invalid axis: {axis:?}"),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Cell index used during iteration
///
/// Instead of storing the cell bounds in the leaf itself, we build them when
/// descending the tree.
///
/// `index` points to where this cell is stored in
/// [`Octree::cells`](super::Octree::cells)
#[derive(Copy, Clone, Debug)]
pub struct CellIndex {
    pub index: usize,
    pub depth: usize,
    pub bounds: CellBounds,
}

impl Default for CellIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CellIndex {
    pub fn new() -> Self {
        CellIndex {
            index: 0,
            bounds: CellBounds::default(),
            depth: 0,
        }
    }

    /// Returns the position of the given corner (0-7)
    ///
    /// Vertices are numbered as follows:
    ///
    /// ```text
    ///         6 -------- 7
    ///        /          /       Z
    ///       / |        / |      ^  _ Y
    ///      4----------5  |      | /
    ///      |  |       |  |      |/
    ///      |  2-------|--3      ---> X
    ///      | /        | /
    ///      |/         |/
    ///      0----------1
    /// ```
    ///
    /// The 8 octree cells are numbered equivalently, based on their corner
    /// vertex.
    pub fn corner(&self, i: Corner) -> (f32, f32, f32) {
        self.bounds.corner(i)
    }

    /// Returns a child cell for the given corner, rooted at the given index
    pub fn child(&self, index: usize, i: Corner) -> Self {
        let bounds = self.bounds.child(i);
        CellIndex {
            index: index + i.index(),
            bounds,
            depth: self.depth + 1,
        }
    }

    /// Converts from a relative position in the cell to an absolute position
    pub fn pos(&self, p: nalgebra::Vector3<u16>) -> nalgebra::Vector3<f32> {
        self.bounds.pos(p)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CellBounds {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl std::ops::Index<Axis> for CellBounds {
    type Output = Interval;

    fn index(&self, axis: Axis) -> &Self::Output {
        match axis {
            X => &self.x,
            Y => &self.y,
            Z => &self.z,
            _ => panic!("invalid axis: {axis:?}"),
        }
    }
}

impl Default for CellBounds {
    fn default() -> Self {
        Self::new()
    }
}

impl CellBounds {
    pub fn new() -> Self {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        Self { x, y, z }
    }

    pub fn corner(&self, i: Corner) -> (f32, f32, f32) {
        let x = if i & X {
            self.x.upper()
        } else {
            self.x.lower()
        };
        let y = if i & Y {
            self.y.upper()
        } else {
            self.y.lower()
        };
        let z = if i & Z {
            self.z.upper()
        } else {
            self.z.lower()
        };
        (x, y, z)
    }

    pub fn child(&self, i: Corner) -> Self {
        let x = if i & X {
            Interval::new(self.x.midpoint(), self.x.upper())
        } else {
            Interval::new(self.x.lower(), self.x.midpoint())
        };
        let y = if i & Y {
            Interval::new(self.y.midpoint(), self.y.upper())
        } else {
            Interval::new(self.y.lower(), self.y.midpoint())
        };
        let z = if i & Z {
            Interval::new(self.z.midpoint(), self.z.upper())
        } else {
            Interval::new(self.z.lower(), self.z.midpoint())
        };
        Self { x, y, z }
    }

    /// Converts from a relative position in the cell to an absolute position
    pub fn pos(&self, p: nalgebra::Vector3<u16>) -> nalgebra::Vector3<f32> {
        let x = self.x.lerp(p.x as f32 / u16::MAX as f32);
        let y = self.y.lerp(p.y as f32 / u16::MAX as f32);
        let z = self.z.lerp(p.z as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
    }

    /// Checks whether the given position is within the cell
    pub fn contains(&self, p: CellVertex) -> bool {
        [X, Y, Z].iter().all(|&i| self[i].contains(p[i]))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cell_encode_decode() {
        for c in [
            Cell::Empty,
            Cell::Invalid,
            Cell::Full,
            Cell::Branch {
                index: 12345,
                thread: 17,
            },
            Cell::Branch {
                index: 0x12340054322345,
                thread: 128,
            },
            Cell::Leaf(Leaf {
                index: 12345,
                mask: 0b101,
            }),
            Cell::Leaf(Leaf {
                index: 0x123400005432,
                mask: 0b11011010,
            }),
            Cell::Leaf(Leaf {
                index: 0x12123400005432,
                mask: 0b11011010,
            }),
        ] {
            assert_eq!(c, Cell::from(CellData::from(c)));
        }
    }

    #[test]
    fn test_cell_corner() {
        let c = Cell::Empty;
        for i in Corner::iter() {
            assert!(!c.corner(i));
        }
        let c = Cell::Full;
        for i in Corner::iter() {
            assert!(c.corner(i));
        }
        let c = Cell::Leaf(Leaf {
            mask: 0b00000010,
            index: 0,
        });
        assert!(!c.corner(Corner::new(0)));
        assert!(c.corner(Corner::new(1)));
    }
}
