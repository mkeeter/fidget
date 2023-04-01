//! Octree construction and meshing

mod builder;
mod cell;
mod dc;
mod frame;
mod gen;
mod octree;
mod output;
mod worker;

#[doc(hidden)]
pub mod types;

// Re-export the main Octree type as public
pub use octree::Octree;

////////////////////////////////////////////////////////////////////////////////

/// An indexed 3D mesh
#[derive(Default, Debug)]
pub struct Mesh {
    /// Triangles, as indexes into [`self.vertices`]
    pub triangles: Vec<nalgebra::Vector3<usize>>,
    /// Vertex positions
    pub vertices: Vec<nalgebra::Vector3<f32>>,
}

impl Mesh {
    /// Builds a new mesh
    pub fn new() -> Self {
        Self::default()
    }
}

/// Settings when building an octree and mesh
#[derive(Copy, Clone, Debug)]
pub struct Settings {
    /// Number of threads to use
    ///
    /// 0 indicates to use the single-threaded evaluator; other values will
    /// spin up _N_ threads to perform octree construction in parallel.
    pub threads: u8,

    /// Minimum depth to recurse in the octree
    pub min_depth: u8,

    /// Maximum depth to recurse in the octree
    pub max_depth: u8,
}
