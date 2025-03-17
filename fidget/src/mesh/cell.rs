//! Data types used in the octree
use crate::types::Interval;

use super::{
    gen::CELL_TO_EDGE_TO_VERT,
    types::{Axis, CellMask, Corner, Edge, Intersection, X, Y, Z},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Cell<const D: usize> {
    Invalid,
    Empty,
    Full,
    Branch {
        /// Index of the next cell in
        /// [`Octree::cells`](super::octree::Octree::cells)
        index: usize,
    },
    Leaf(Leaf<D>),
}

impl<const D: usize> Cell<D> {
    /// Checks whether the given corner is empty (`false`) or full (`true`)
    ///
    /// # Panics
    /// If the cell is a branch or invalid
    pub fn corner(self, c: Corner<D>) -> bool {
        match self {
            Cell::Leaf(Leaf { mask, .. }) => mask & c,
            Cell::Empty => false,
            Cell::Full => true,
            Cell::Branch { .. } | Cell::Invalid => panic!(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Leaf<const D: usize> {
    /// Mask of corner occupancy
    pub mask: CellMask<D>,

    /// Index of first vertex in [`Octree::verts`](super::octree::Octree::verts)
    pub index: usize,
}

impl<const D: usize> Leaf<D> {
    /// Returns the edge intersection for the given edge (if present)
    pub fn edge(&self, e: Edge) -> Option<Intersection> {
        CELL_TO_EDGE_TO_VERT[self.mask.index()][e.index()]
    }
}

// TODO make this generic across dimensions?
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

impl std::ops::Index<Axis<3>> for CellVertex {
    type Output = f32;

    fn index(&self, axis: Axis<3>) -> &Self::Output {
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
    pub fn corner(&self, i: Corner<3>) -> (f32, f32, f32) {
        self.bounds.corner(i)
    }

    /// Returns a child cell for the given corner, rooted at the given index
    pub fn child(&self, index: usize, i: Corner<3>) -> Self {
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

// TODO make this generic across dimensions?
#[derive(Copy, Clone, Debug)]
pub struct CellBounds {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl std::ops::Index<Axis<3>> for CellBounds {
    type Output = Interval;

    fn index(&self, axis: Axis<3>) -> &Self::Output {
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

    pub fn corner(&self, i: Corner<3>) -> (f32, f32, f32) {
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

    pub fn child(&self, i: Corner<3>) -> Self {
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
    fn test_cell_corner() {
        let c = Cell::<3>::Empty;
        for i in Corner::iter() {
            assert!(!c.corner(i));
        }
        let c = Cell::<3>::Full;
        for i in Corner::iter() {
            assert!(c.corner(i));
        }
        let c = Cell::<3>::Leaf(Leaf {
            mask: CellMask::new(0b00000010),
            index: 0,
        });
        assert!(!c.corner(Corner::new(0)));
        assert!(c.corner(Corner::new(1)));
    }
}
