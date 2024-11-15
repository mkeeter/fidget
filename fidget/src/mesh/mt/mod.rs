//! Multithreaded implementations
mod dc;
mod octree;
mod pool;

pub use dc::DcWorker;
pub use octree::OctreeWorker;

/// Strong type for multithreaded settings
pub(crate) struct MultithreadedSettings {
    pub depth: u8,
    pub threads: std::num::NonZeroUsize,
}
