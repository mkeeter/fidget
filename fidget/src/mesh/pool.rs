//! Minimal utilities for thread pooling
use std::sync::atomic::{AtomicUsize, Ordering};

/// Stores data used to synchronize a thread pool
pub struct ThreadPool {
    threads: std::sync::RwLock<Vec<std::thread::Thread>>,
    counter: AtomicUsize,
}

impl ThreadPool {
    /// Builds thread pool storage for `n` threads
    pub fn new(n: usize) -> Self {
        Self {
            threads: std::sync::RwLock::new(vec![std::thread::current(); n]),
            counter: AtomicUsize::new(0),
        }
    }

    /// Builds a `ThreadContext` for the given thread
    ///
    /// This must be called in a different thread than the one used to build the
    /// thread pool (the latter is assumed to be the main application thread;
    /// the former is a worker thread).
    ///
    /// # Panics
    /// If `index` is greater than the `n` used in the constructor, or if this
    /// is called from the same thread used to build the thread pool.
    pub fn start(&self, index: usize) -> ThreadContext {
        // Record our current index
        let mut w = self.threads.write().unwrap();
        let thread_count = w.len();
        assert!(index < thread_count);
        let my_thread = std::thread::current();
        assert_ne!(my_thread.id(), w[index].id());
        w[index] = my_thread;
        self.counter.fetch_add(1, Ordering::Release);

        // Wake all of the other workers; if everyone has registered themselves,
        // then the counter will be at thread_count and everyone will continue.
        for (i, t) in w.iter().enumerate() {
            if i != index {
                t.unpark();
            }
        }
        drop(w);

        // Wait until every thread has installed itself into the array
        while self.counter.load(Ordering::Acquire) < thread_count {
            std::thread::park();
        }

        let threads = self.threads.read().unwrap();
        ThreadContext {
            threads,
            counter: &self.counter,
            index,
        }
    }
}

/// Local context for a thread operating within a pool
pub struct ThreadContext<'a> {
    threads: std::sync::RwLockReadGuard<'a, Vec<std::thread::Thread>>,
    counter: &'a AtomicUsize,
    index: usize,
}

impl ThreadContext<'_> {
    /// If some threads in the pool are sleeping, wakes them up
    ///
    /// This function should be called when work is available.
    pub fn wake(&self) {
        if self.counter.load(Ordering::Acquire) >> 8 != 0 {
            for (i, t) in self.threads.iter().enumerate() {
                if i != self.index {
                    t.unpark();
                }
            }
        }
    }

    /// Sends the given thread to sleep
    ///
    /// Returns `true` on success; `false` if all work is done and the thread
    /// should now exit.
    pub fn sleep(&self) -> bool {
        // At this point, the thread doesn't have any work to do, so we'll
        // consider putting it to sleep.  However, if every other thread is
        // sleeping, then we're ready to exit; we'll wake them all up.
        let c = 1 + (self.counter.fetch_add(256, Ordering::Release) >> 8);
        if c == self.threads.len() {
            // Wake up the other threads, so they notice that we're done
            for (i, t) in self.threads.iter().enumerate() {
                if i != self.index {
                    t.unpark();
                }
            }
            return false;
        }
        // There are other active threads, so park ourselves and wait for
        // someone else to wake us up.
        std::thread::park();
        if self.counter.load(Ordering::Acquire) >> 8 == self.threads.len() {
            return false;
        }
        // Back to the grind
        self.counter.fetch_sub(256, Ordering::Release);
        true
    }
}
