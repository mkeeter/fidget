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

#[derive(Copy, Clone, Debug)]
pub struct CellVertex<const D: usize> {
    /// Position of this vertex
    pub pos: nalgebra::OVector<f32, nalgebra::Const<D>>,
}

impl<const D: usize> Default for CellVertex<D> {
    fn default() -> Self {
        Self {
            pos: nalgebra::OVector::<f32, nalgebra::Const<D>>::from_element(
                f32::NAN,
            ),
        }
    }
}

impl<const D: usize> std::ops::Index<Axis<D>> for CellVertex<D> {
    type Output = f32;

    fn index(&self, axis: Axis<D>) -> &Self::Output {
        &self.pos[axis.index()]
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
    pub bounds: CellBounds<3>,
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

#[derive(Copy, Clone, Debug)]
pub struct CellBounds<const D: usize> {
    pub bounds: [Interval; D],
}

impl<const D: usize> std::ops::Index<Axis<D>> for CellBounds<D> {
    type Output = Interval;

    fn index(&self, axis: Axis<D>) -> &Self::Output {
        &self.bounds[axis.index()]
    }
}

impl<const D: usize> Default for CellBounds<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> CellBounds<D> {
    pub fn new() -> Self {
        Self {
            bounds: [Interval::new(-1.0, 1.0); D],
        }
    }

    /// Checks whether the given position is within the cell
    pub fn contains(&self, p: CellVertex<D>) -> bool {
        (0..D as u8)
            .map(|i| Axis::<D>::new(i))
            .all(|i| self[i].contains(p[i]))
    }
}

impl CellBounds<3> {
    pub fn corner(&self, i: Corner<3>) -> (f32, f32, f32) {
        let x = if i & X {
            self.bounds[0].upper()
        } else {
            self.bounds[0].lower()
        };
        let y = if i & Y {
            self.bounds[1].upper()
        } else {
            self.bounds[1].lower()
        };
        let z = if i & Z {
            self.bounds[2].upper()
        } else {
            self.bounds[2].lower()
        };
        (x, y, z)
    }

    pub fn child(&self, i: Corner<3>) -> Self {
        let x = if i & X {
            Interval::new(self.bounds[0].midpoint(), self.bounds[0].upper())
        } else {
            Interval::new(self.bounds[0].lower(), self.bounds[0].midpoint())
        };
        let y = if i & Y {
            Interval::new(self.bounds[1].midpoint(), self.bounds[1].upper())
        } else {
            Interval::new(self.bounds[1].lower(), self.bounds[1].midpoint())
        };
        let z = if i & Z {
            Interval::new(self.bounds[2].midpoint(), self.bounds[2].upper())
        } else {
            Interval::new(self.bounds[2].lower(), self.bounds[2].midpoint())
        };
        Self { bounds: [x, y, z] }
    }

    /// Converts from a relative position in the cell to an absolute position
    pub fn pos(&self, p: nalgebra::Vector3<u16>) -> nalgebra::Vector3<f32> {
        let x = self.bounds[0].lerp(p.x as f32 / u16::MAX as f32);
        let y = self.bounds[1].lerp(p.y as f32 / u16::MAX as f32);
        let z = self.bounds[2].lerp(p.z as f32 / u16::MAX as f32);
        nalgebra::Vector3::new(x, y, z)
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
