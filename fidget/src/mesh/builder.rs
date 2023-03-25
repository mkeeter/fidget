use super::{
    cell::{CellIndex, CellVertex},
    Mesh,
};

/// Container used during construction of a [`Mesh`]
#[derive(Default)]
pub struct MeshBuilder {
    /// Map from indexes in [`Octree::verts`] to `out.vertices`
    ///
    /// `usize::MAX` is used a marker for an unmapped vertex
    map: Vec<usize>,
    out: Mesh,
}

impl MeshBuilder {
    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`].
    pub fn get(
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
    pub fn push(&mut self, tri: nalgebra::Vector3<usize>) {
        self.out.triangles.push(tri)
    }
    pub fn take(self) -> Mesh {
        self.out
    }
}
