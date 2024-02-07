//! Octree construction and meshing
//!
//! This module implements
//! [Manifold Dual Contouring](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf),
//! to generate a triangle mesh from an implicit surface (or anything
//! implementing [`Shape`](crate::eval::Shape)).
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
//!     eval::MathShape,
//!     mesh::{Octree, Settings},
//!     vm::VmShape
//! };
//!
//! let (node, ctx) = fidget::rhai::eval("sphere(0, 0, 0, 0.6).call(x, y, z)")?;
//! let shape = VmShape::new(&ctx, node)?;
//! let settings = Settings {
//!     threads: 8,
//!     min_depth: 4,
//!     max_depth: 4,
//!     bounds: Default::default(),
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
mod dc;
mod fixup;
mod frame;
mod gen;
mod mt;
mod octree;
mod output;
mod qef;

#[doc(hidden)]
pub mod types;

// Re-export CellBounds and the main Octree type as public
pub use cell::CellBounds;
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
    /// Number of threads to use
    ///
    /// 0 indicates to use the single-threaded evaluator; other values will
    /// spin up _N_ threads to perform octree construction in parallel.
    pub threads: u8,

    /// Minimum depth to recurse in the octree
    pub min_depth: u8,

    /// Maximum depth to recurse in the octree
    ///
    /// If this is `> min_depth`, then after the octree is initially built
    /// (recursing to `min_depth`), cells with escaped vertices are subdivided
    /// recursively up to a limit of `max_depth`.
    ///
    /// This is **much slower**.
    pub max_depth: u8,

    /// Bounds to evaluate the octree in
    pub bounds: CellBounds,
}
