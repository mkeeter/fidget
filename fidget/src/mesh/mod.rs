//! Octree construction and meshing
//!
//! This module implements
//! [Manifold Dual Contouring](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf),
//! to generate a triangle mesh from an implicit surface (or anything
//! implementing [`Shape`](crate::shape::Shape)).
//!
//! The resulting meshes should be
//! - Manifold
//! - Watertight
//! - Preserving sharp features (corners / edges)
//!
//! However, they may contain self-intersections, and are not guaranteed to
//! catch thin features (below the sampling grid resolution).
//!
//! The resulting [`Mesh`] objects can be written out as STL files.
//!
//! Here's a full example:
//!
//! ```
//! use fidget::{
//!     context::Tree,
//!     mesh::{Octree, Settings},
//!     vm::VmShape
//! };
//!
//! let tree: Tree = fidget::rhai::engine()
//!     .eval("sphere(#{ center: [0, 0, 0], radius: 0.6 })")?;
//! let shape = VmShape::from(tree);
//! let settings = Settings {
//!     depth: 4,
//!     ..Default::default()
//! };
//! let o = Octree::build(&shape, settings);
//! let mesh = o.walk_dual(settings);
//!
//! // Open a file to write, e.g.
//! // let mut f = std::fs::File::create("out.stl")?;
//! # let mut f = vec![];
//! mesh.write_stl(&mut f)?;
//! # Ok::<(), fidget::Error>(())
//! ```

mod builder;
mod cell;
mod codegen;
mod dc;
mod frame;
mod octree;
mod output;
mod qef;

use crate::render::{ThreadPool, View3};

#[doc(hidden)]
pub mod types;

// Re-export the main Octree type as public
pub use octree::Octree;

////////////////////////////////////////////////////////////////////////////////

/// An indexed 3D mesh
#[derive(Default, Debug)]
pub struct Mesh {
    /// Triangles, as indexes into [`self.vertices`](Self::vertices)
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
#[derive(Copy, Clone)]
pub struct Settings<'a> {
    /// Depth to recurse in the octree
    pub depth: u8,

    /// Viewport to provide a world-to-model transform
    pub view: View3,

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<&'a ThreadPool>,
}

impl Default for Settings<'_> {
    fn default() -> Self {
        Self {
            depth: 3,
            view: Default::default(),
            threads: Some(&ThreadPool::Global),
        }
    }
}
