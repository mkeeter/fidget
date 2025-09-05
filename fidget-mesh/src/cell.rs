//! Data types used in the octree
use fidget_core::types::Interval;

use super::{
    codegen::CELL_TO_EDGE_TO_VERT,
    types::{Axis, CellMask, Corner, Edge, Intersection},
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
pub struct CellIndex<const D: usize> {
    /// Cell index in `Octree::cells`; `None` is the root
    pub index: Option<(usize, u8)>,
    pub depth: usize,
    pub bounds: CellBounds<D>,
}

impl<const D: usize> Default for CellIndex<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> CellIndex<D> {
    pub fn new() -> Self {
        CellIndex {
            index: None,
            bounds: CellBounds::default(),
            depth: 0,
        }
    }

    /// Returns a child cell for the given corner, rooted at the given index
    pub fn child(&self, index: usize, i: Corner<D>) -> Self {
        let bounds = self.bounds.child(i);
        CellIndex {
            index: Some((index, i.get())),
            bounds,
            depth: self.depth + 1,
        }
    }

    /// Returns the position of the given corner
    ///
    /// Vertices are numbered as follows in 3D:
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
    ///
    /// In 2D, only corners on the XY plane (0-4) are valid.
    pub fn corner(&self, i: Corner<D>) -> [f32; D] {
        self.bounds.corner(i)
    }
}

impl CellIndex<3> {
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
        Axis::array()
            .into_iter()
            .all(|axis| self[axis].contains(p[axis]))
    }

    pub fn child(&self, corner: Corner<D>) -> Self {
        let bounds = Axis::array().map(|axis| {
            let i = axis.index();
            if corner & axis {
                Interval::new(self.bounds[i].midpoint(), self.bounds[i].upper())
            } else {
                Interval::new(self.bounds[i].lower(), self.bounds[i].midpoint())
            }
        });
        Self { bounds }
    }

    pub fn corner(&self, corner: Corner<D>) -> [f32; D] {
        Axis::array().map(|axis| {
            let i = axis.index();
            if corner & axis {
                self.bounds[i].upper()
            } else {
                self.bounds[i].lower()
            }
        })
    }

    /// Converts from a relative position in the cell to an absolute position
    pub fn pos(
        &self,
        p: nalgebra::OVector<u16, nalgebra::Const<D>>,
    ) -> nalgebra::OVector<f32, nalgebra::Const<D>> {
        let mut out = nalgebra::OVector::<f32, nalgebra::Const<D>>::zeros();
        for i in 0..D {
            out[i] = self.bounds[i].lerp(p[i] as f32 / u16::MAX as f32);
        }
        out
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
