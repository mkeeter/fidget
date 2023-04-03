//! Mesh builder data structure and implementation
use super::{
    cell::{CellIndex, CellVertex},
    dc::{self, DcBuilder},
    frame::Frame,
    Mesh, Octree,
};

/// Container used during construction of a [`Mesh`]
#[derive(Default)]
pub struct MeshBuilder {
    /// Map from indexes in [`Octree::verts`](super::Octree::verts) to
    /// `out.vertices`
    ///
    /// `usize::MAX` is used a marker for an unmapped vertex
    map: Vec<usize>,
    out: Mesh,
}

impl MeshBuilder {
    pub fn take(self) -> Mesh {
        self.out
    }
}

impl DcBuilder for MeshBuilder {
    fn cell(&mut self, octree: &Octree, cell: CellIndex) {
        dc::dc_cell(octree, cell, self);
    }
    fn face<F: Frame>(&mut self, octree: &Octree, a: CellIndex, b: CellIndex) {
        dc::dc_face::<F, _>(octree, a, b, self)
    }
    fn edge<F: Frame>(
        &mut self,
        octree: &Octree,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
    ) {
        dc::dc_edge::<F, _>(octree, a, b, c, d, self)
    }
    fn triangle(&mut self, a: usize, b: usize, c: usize) {
        self.out.triangles.push(nalgebra::Vector3::new(a, b, c))
    }

    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`](super::Octree::verts).
    fn vertex(
        &mut self,
        v: usize,
        cell: CellIndex,
        verts: &[CellVertex],
    ) -> usize {
        if v >= self.map.len() {
            self.map.resize(v + 1, usize::MAX);
        }
        match self.map[v] {
            usize::MAX => {
                let next_vert = self.out.vertices.len();
                self.out.vertices.push(cell.pos(verts[v]));
                self.map[v] = next_vert;

                next_vert
            }
            u => u,
        }
    }
}
