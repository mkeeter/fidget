//! Octree cells
use crate::eval::types::Interval;

use super::{
    gen::CELL_TO_EDGE_TO_VERT,
    types::{Corner, Edge, Intersection, X, Y, Z},
};

/// Raw cell data
///
/// Unpack to a [`Cell`] to actually use it
#[derive(Copy, Clone)]
pub struct CellData(u64);

impl From<Cell> for CellData {
    fn from(c: Cell) -> Self {
        let i = match c {
            Cell::Empty => 0b00 << 62,
            Cell::Full => 0b01 << 62,
            Cell::Branch { index } => {
                debug_assert!(index < (1 << 62));
                0b10 << 62 | index as u64
            }
            Cell::Leaf(Leaf { mask, index }) => {
                debug_assert!(index < (1 << 54));
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

impl CellData {
    pub fn new(i: u64) -> Self {
        Self(i)
    }
}

static_assertions::const_assert_eq!(
    std::mem::size_of::<usize>(),
    std::mem::size_of::<u64>()
);

/// Unpacked form of [`CellData`]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Cell {
    Empty,
    Full,
    Branch { index: usize },
    Leaf(Leaf),
}

impl From<CellData> for Cell {
    fn from(c: CellData) -> Self {
        let i = c.0 as usize;
        match (i >> 62) & 0b11 {
            0b00 => Cell::Empty,
            0b01 => Cell::Full,
            0b10 => Cell::Branch {
                index: i & ((1 << 62) - 1),
            },
            0b11 => Cell::Leaf(Leaf {
                mask: (i >> 54) as u8,
                index: i & ((1 << 54) - 1),
            }),
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Leaf {
    pub mask: u8,
    pub index: usize,
}

impl Leaf {
    pub fn edge(&self, e: Edge) -> Intersection {
        let out = CELL_TO_EDGE_TO_VERT[self.mask as usize][e.index()];
        debug_assert_ne!(out.vert.0, u8::MAX);
        debug_assert_ne!(out.edge.0, u8::MAX);
        out
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CellVertex {
    /// Position, as a relative offset within a cell's bounding box
    pub pos: nalgebra::Vector3<u16>,

    /// If a vertex is valid, its original position was within the bounding box
    pub _valid: bool,
}

impl From<CellVertex> for nalgebra::Vector3<u16> {
    fn from(v: CellVertex) -> Self {
        v.pos
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Cell index used during iteration
///
/// Instead of storing the cell bounds in the leaf itself, we build them when
/// descending the tree.
///
/// `index` points to where this cell is stored in [`Octree::cells`]
#[derive(Copy, Clone, Debug)]
pub struct CellIndex {
    pub index: usize,
    pub depth: usize,
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl Default for CellIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CellIndex {
    pub fn new() -> Self {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        CellIndex {
            index: 0,
            x,
            y,
            z,
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

    /// Returns the interval of the given child (0-7)
    pub fn interval(&self, i: Corner) -> (Interval, Interval, Interval) {
        // TODO: make this a function in `Interval`?
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
        (x, y, z)
    }

    /// Converts from a relative position in the cell to an absolute position
    pub fn pos<P: Into<nalgebra::Vector3<u16>>>(
        &self,
        p: P,
    ) -> nalgebra::Vector3<f32> {
        let p = p.into();
        let x = self.x.lerp(p.x as f32 / u16::MAX as f32);
        let y = self.y.lerp(p.y as f32 / u16::MAX as f32);
        let z = self.z.lerp(p.z as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
    }

    /// Converts from an absolute position to a relative position in the cell
    ///
    /// The `bool` indicates whether the vertex was clamped into the cell's
    /// bounding box.
    pub fn relative(&self, p: nalgebra::Vector3<f32>) -> CellVertex {
        let x = (p.x - self.x.lower()) / self.x.width() * u16::MAX as f32;
        let y = (p.y - self.y.lower()) / self.y.width() * u16::MAX as f32;
        let z = (p.z - self.z.lower()) / self.z.width() * u16::MAX as f32;

        let valid = x >= 0.0
            && x <= u16::MAX as f32
            && y >= 0.0
            && y <= u16::MAX as f32
            && z >= 0.0
            && z <= u16::MAX as f32;

        CellVertex {
            pos: nalgebra::Vector3::new(
                x.clamp(0.0, u16::MAX as f32) as u16,
                y.clamp(0.0, u16::MAX as f32) as u16,
                z.clamp(0.0, u16::MAX as f32) as u16,
            ),
            _valid: valid,
        }
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
            Cell::Full,
            Cell::Branch { index: 12345 },
            Cell::Branch {
                index: 0x1234000054322345,
            },
            Cell::Leaf(Leaf {
                index: 12345,
                mask: 0b101,
            }),
            Cell::Leaf(Leaf {
                index: 0x123400005432,
                mask: 0b11011010,
            }),
        ] {
            assert_eq!(c, Cell::from(CellData::from(c)));
        }
    }
}
