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

////////////////////////////////////////////////////////////////////////////////

/// Queue for use in a thread pool
///
/// This queue contains a local queue plus references to other threads' queues,
/// so that we can steal items if our queue runs dry.
pub struct QueuePool<T> {
    /// Our personal queue of tasks to complete
    ///
    /// Other threads may steal from this queue!
    queue: crossbeam_deque::Worker<T>,

    /// Queues from which we can steal other workers' tasks
    ///
    /// Our own queue is at index `self.index` in this list and is skipped when
    /// attempting to steal an item.
    friend_queue: Vec<crossbeam_deque::Stealer<T>>,

    /// Marks whether the queue has received new items since the last `pop`
    changed: bool,

    /// Index of this queue within the pool.
    index: usize,
}

impl<T> QueuePool<T> {
    /// Builds a new set of queues for `n` threads
    pub fn new(n: usize) -> Vec<Self> {
        let task_queues = (0..n)
            .map(|_| crossbeam_deque::Worker::<T>::new_lifo())
            .collect::<Vec<_>>();

        let stealers =
            task_queues.iter().map(|t| t.stealer()).collect::<Vec<_>>();

        task_queues
            .into_iter()
            .enumerate()
            .map(|(index, queue)| Self {
                queue,
                friend_queue: stealers.clone(),
                changed: false,
                index,
            })
            .collect()
    }

    /// Pops an item from this queue or steals from another
    ///
    /// Returns the item along with its source index.
    ///
    /// Sets `self.changed` to `false`
    pub fn pop(&mut self) -> Option<(T, usize)> {
        self.changed = false;
        self.queue.pop().map(|v| (v, self.index)).or_else(|| {
            // Try stealing from all of our friends
            use crossbeam_deque::Steal;
            for i in 1..self.friend_queue.len() {
                let i = (i + self.index) % self.friend_queue.len();
                let q = &self.friend_queue[i];
                loop {
                    match q.steal() {
                        Steal::Success(v) => return Some((v, i)),
                        Steal::Empty => break,
                        Steal::Retry => continue,
                    }
                }
            }
            None
        })
    }

    /// Pushes an item to this queue, setting `self.changed` to true
    pub fn push(&mut self, t: T) {
        self.queue.push(t);
        self.changed = true;
    }

    /// Returns the value of `self.changed`
    ///
    /// This indicates whether we have pushed items to the queue since the last
    /// call to `pop()`.
    pub fn changed(&self) -> bool {
        self.changed
    }
}
