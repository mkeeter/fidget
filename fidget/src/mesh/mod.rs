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
//!     mesh::{Octree, Settings},
//!     vm::VmShape
//! };
//!
//! let tree = fidget::rhai::eval("sphere(0, 0, 0, 0.6)")?;
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
mod dc;
mod frame;
mod gen;
mod octree;
mod output;
mod qef;

use crate::render::View3;

/// Number of threads to use during evaluation
///
/// In a WebAssembly build, only the [`ThreadCount::One`] variant is available.
#[derive(Copy, Clone, Debug)]
pub enum ThreadCount {
    /// Perform all evaluation in the main thread, not spawning any workers
    One,

    /// Spawn some number of worker threads for evaluation
    ///
    /// This can be set to `1`, in which case a single worker thread will be
    /// spawned; this is different from doing work in the main thread, but not
    /// particularly useful!
    #[cfg(not(target_arch = "wasm32"))]
    Many(std::num::NonZeroUsize),
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::num::NonZeroUsize> for ThreadCount {
    fn from(v: std::num::NonZeroUsize) -> Self {
        match v.get() {
            0 => unreachable!(),
            1 => ThreadCount::One,
            _ => ThreadCount::Many(v),
        }
    }
}

/// Single-threaded mode is shown as `-`; otherwise, an integer
impl std::fmt::Display for ThreadCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadCount::One => write!(f, "-"),
            #[cfg(not(target_arch = "wasm32"))]
            ThreadCount::Many(n) => write!(f, "{n}"),
        }
    }
}

impl ThreadCount {
    /// Gets the thread count
    ///
    /// Returns `None` if we are required to be single-threaded
    pub fn get(&self) -> Option<usize> {
        match self {
            ThreadCount::One => None,
            #[cfg(not(target_arch = "wasm32"))]
            ThreadCount::Many(v) => Some(v.get()),
        }
    }
}

impl Default for ThreadCount {
    #[cfg(target_arch = "wasm32")]
    fn default() -> Self {
        Self::One
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn default() -> Self {
        Self::Many(std::num::NonZeroUsize::new(8).unwrap())
    }
}

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

    /// Viewport to provide a world-to-model transform
    pub view: View3,

    /// Number of threads to use
    ///
    /// 1 indicates to use the single-threaded evaluator; other values will
    /// spin up _N_ threads to perform octree construction in parallel.
    pub threads: ThreadCount,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            depth: 3,
            view: Default::default(),
            threads: ThreadCount::default(),
        }
    }
}
