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
            phase: 1,
        }
    }
}

/// Local context for a thread operating within a pool
pub struct ThreadContext<'a> {
    threads: std::sync::RwLockReadGuard<'a, Vec<std::thread::Thread>>,
    counter: &'a AtomicUsize,
    index: usize,

    /// We operate in 4 phases, depending on the value of `phase % 4`:
    ///
    /// `phase` | byte | direction | start | end
    /// --------|------|-----------|-------|-----
    ///    0    |   0  |  up       | 0     | `N`
    ///    1    |   1  |  up       | 0     | `N`
    ///    2    |   0  |  down     | `N`   | 0
    ///    3    |   1  |  down     | `N`   | 0
    ///
    /// (`N` is `self.threads.len()`)
    ///
    /// Note that the pairs of adjacent phases are non-interfering: if a thread
    /// in phase 0 notices that it has hit `N`, then it can immediately enter
    /// phase 1 and start modifing byte 1 of the counter, without other threads
    /// in phase 0 noticing.
    phase: u8,
}

impl ThreadContext<'_> {
    /// If some threads in the pool are sleeping, wakes them up
    ///
    /// This function should be called when work is available.
    pub fn wake(&self) {
        let mut v = self.counter.load(Ordering::Acquire);
        if (self.phase % 2) == 1 {
            v >>= 8;
        }
        // Check to see if any other threads are sleeping; otherwise, skip the
        // unparking step (because it costs time)
        if (((self.phase % 2) / 2 == 0) && v != 0)
            || (((self.phase % 2) / 2 == 1) && v != self.threads.len())
        {
            for (i, t) in self.threads.iter().enumerate() {
                if i != self.index {
                    t.unpark();
                }
            }
        }
    }

    /// Sends the given thread to sleep
    ///
    /// Returns `true` if the thread should continue running; `false` if all
    /// threads in the pool have requested to sleep, indicating that all work is
    /// done and they should now halt.
    pub fn sleep(&mut self) -> bool {
        // At this point, the thread doesn't have any work to do, so we'll
        // consider putting it to sleep.  However, if every other thread is
        // sleeping, then we're ready to exit; we'll wake them all up.
        let n = self.threads.len();
        let mut done = match self.phase % 4 {
            0 => (1 + self.counter.fetch_add(1, Ordering::Release)) & 0xFF == n,
            1 => 1 + (self.counter.fetch_add(256, Ordering::Release) >> 8) == n,
            2 => (self.counter.fetch_sub(1, Ordering::Release) - 1) & 0xFF == 0,
            3 => (self.counter.fetch_sub(256, Ordering::Release) >> 8) - 1 == 0,
            _ => unreachable!(),
        };

        if done {
            // Wake up the other threads, so they notice that we're done
            for (i, t) in self.threads.iter().enumerate() {
                if i != self.index {
                    t.unpark();
                }
            }
        } else {
            // There are other active threads, so park ourselves and wait for
            // someone else to wake us up.
            std::thread::park();

            // Someone has woken us up!  Check our counter and see whether we've
            // been woken up because every thread has finished.
            let c = self.counter.load(Ordering::Acquire);
            done = match self.phase % 4 {
                0 => c & 0xFF == n,
                1 => c >> 8 == n,
                2 => c & 0xFF == 0,
                3 => c >> 8 == 0,
                _ => unreachable!(),
            };
            if !done {
                // Back to the grind
                match self.phase % 4 {
                    0 => self.counter.fetch_sub(1, Ordering::Release),
                    1 => self.counter.fetch_sub(256, Ordering::Release),
                    2 => self.counter.fetch_add(1, Ordering::Release),
                    3 => self.counter.fetch_add(256, Ordering::Release),
                    _ => unreachable!(),
                };
            }
        }

        if done {
            self.phase += 1;
        }
        !done
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn queue_pool() {
        let mut queues = QueuePool::new(2);
        let mut counters = [0i32; 2];
        const DEPTH: usize = 5;
        queues[0].push(DEPTH);

        // Confirm that stealing leads to shared work between two threads
        std::thread::scope(|s| {
            for (q, c) in queues.iter_mut().zip(counters.iter_mut()) {
                s.spawn(|| {
                    while let Some((i, _)) = q.pop() {
                        *c += 1;
                        if i != 0 {
                            q.push(i - 1);
                            q.push(i - 1);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                });
            }
        });
        assert_eq!(
            counters[0] + counters[1],
            (1 << (DEPTH + 1)) - 1,
            "threads did not complete all work"
        );
        assert_eq!(
            counters[0].abs_diff(counters[1]),
            1,
            "unequal work distribution between threads"
        );
    }

    #[test]
    fn thread_ctx() {
        const N: usize = 8;
        let pool = &ThreadPool::new(N);
        let done = &AtomicUsize::new(0);

        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if done.load(Ordering::Acquire) != N {
                    eprintln!("deadlock in `thread_ctx` test; aborting");
                    std::process::exit(1);
                }
            });
            for i in 0..N {
                s.spawn(move || {
                    let mut ctx = pool.start(i);
                    let t = std::time::Duration::from_millis(1);
                    for _ in 0..8 {
                        for _ in 0..i {
                            std::thread::sleep(t);
                            ctx.wake();
                        }
                        while ctx.sleep() {
                            // Loop forever
                        }
                    }
                    done.fetch_add(1, Ordering::Release);
                });
            }
        });
        assert_eq!(done.load(Ordering::Acquire), N);
    }
}
