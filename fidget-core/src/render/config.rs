//! Types used in configuration structures
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Thread pool to use for multithreaded rendering
///
/// Most users will use the global Rayon pool, but it's possible to provide your
/// own as well.
pub enum ThreadPool {
    /// User-provided pool
    Custom(rayon::ThreadPool),
    /// Global Rayon pool
    Global,
}

impl ThreadPool {
    /// Runs a function across the thread pool
    pub fn run<F: FnOnce() -> V + Send, V: Send>(&self, f: F) -> V {
        match self {
            ThreadPool::Custom(p) => p.install(f),
            ThreadPool::Global => f(),
        }
    }

    /// Returns the number of threads in the pool
    pub fn thread_count(&self) -> usize {
        match self {
            ThreadPool::Custom(p) => p.current_num_threads(),
            ThreadPool::Global => rayon::current_num_threads(),
        }
    }
}

/// Token to cancel an in-progress operation
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Build a new token, which is initialize as "not cancelled"
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark this token as cancelled
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Check if the token is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Returns a raw pointer to the inner flag
    ///
    /// This is used in shared memory environments where the `CancelToken`
    /// itself cannot be passed between threads, i.e. to send a cancel token to
    /// a web worker.
    ///
    /// To avoid a memory leak, the pointer must be converted back to a
    /// `CancelToken` using [`CancelToken::from_raw`].  In the meantime, users
    /// should refrain from writing to the raw pointer.
    #[doc(hidden)]
    pub fn into_raw(self) -> *const AtomicBool {
        Arc::into_raw(self.0)
    }

    /// Reclaims a released cancel token pointer
    ///
    /// # Safety
    /// The pointer must have been previously returned by a call to
    /// [`CancelToken::into_raw`].
    #[doc(hidden)]
    pub unsafe fn from_raw(ptr: *const AtomicBool) -> Self {
        let a = unsafe { Arc::from_raw(ptr) };
        Self(a)
    }
}
