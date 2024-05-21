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
//!     shape::MathShape,
//!     mesh::{Octree, Settings},
//!     vm::VmShape
//! };
//!
//! let tree = fidget::rhai::eval("sphere(0, 0, 0, 0.6)")?;
//! let shape = VmShape::from_tree(&tree);
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

use crate::shape::Bounds;

mod builder;
mod cell;
mod dc;
mod frame;
mod gen;
mod octree;
mod output;
mod qef;

#[cfg(not(target_arch = "wasm32"))]
mod mt;

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
#[derive(Copy, Clone, Debug)]
pub struct Settings {
    /// Depth to recurse in the octree
    pub depth: u8,

    /// Bounds for meshing
    pub bounds: Bounds<3>,

    /// Number of threads to use
    ///
    /// 1 indicates to use the single-threaded evaluator; other values will
    /// spin up _N_ threads to perform octree construction in parallel.
    #[cfg(not(target_arch = "wasm32"))]
    pub threads: std::num::NonZeroUsize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            depth: 3,
            bounds: Default::default(),

            #[cfg(not(target_arch = "wasm32"))]
            threads: std::num::NonZeroUsize::new(8).unwrap(),
        }
    }
}

impl Settings {
    #[cfg(not(target_arch = "wasm32"))]
    fn threads(&self) -> usize {
        self.threads.get()
    }

    #[cfg(target_arch = "wasm32")]
    fn threads(&self) -> usize {
        1
    }
}
