use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc::TryRecvError,
    Arc,
};

use super::{
    cell::{Cell, CellData, CellIndex},
    octree::{CellResult, EvalData, EvalGroup, EvalStorage},
    types::Corner,
    Octree, Settings,
};
use crate::eval::Family;

/// Represents a chunk of work that should be handled by a worker
///
/// Specifically, the work is to subdivide the given cell and evaluate its 8
/// octants, sending results back to the parent (which is numbered implicitly
/// based on what queue we stole this from).
struct Task<I: Family> {
    eval: Arc<EvalGroup<I>>,

    /// Parent cell, which must be a branch cell pointing to the given index
    parent: CellIndex,

    /// Beginning of the 8x child region in the parent octree
    index: usize,
}

#[derive(Debug)]
struct Done {
    /// The cell index within the parent octree, to which we're sending data
    cell_index: usize,

    /// The resulting cell
    ///
    /// If this is filled / empty, then it's owned by the receiving octree;
    /// otherwise, this is a leaf or branch and may point to one of the other
    /// worker's data arrays.
    child: CellData,
}

pub struct Worker<I: Family> {
    thread_index: usize,
    octree: Octree,
    queue: crossbeam_deque::Worker<Task<I>>,
    done: std::sync::mpsc::Receiver<Done>,

    /// Queues from which we can steal other workers' tasks
    ///
    /// Each task has `thread_count` friend queues, including its own; it would
    /// be silly to steal from your own queue, but that keeps the code cleaner.
    friend_queue: Vec<crossbeam_deque::Stealer<Task<I>>>,

    /// When a worker finishes a task, it returns it through these queues
    ///
    /// Like `friend_queue`, there's one per thread, including the worker's own
    /// thread; it would be silly to send stuff back to your own thread via the
    /// queue (rather than storing it directly).
    friend_done: Vec<std::sync::mpsc::Sender<Done>>,

    /// Per-thread local data for evaluation, to avoid allocation churn
    data: EvalData<I>,

    /// Per-thread local storage for evaluators, to avoid allocation churn
    storage: EvalStorage<I>,
}

impl<I: Family> Worker<I> {
    pub fn scheduler(eval: Arc<EvalGroup<I>>, settings: Settings) -> Octree {
        let task_queues = (0..settings.threads)
            .map(|_| crossbeam_deque::Worker::<Task<I>>::new_lifo())
            .collect::<Vec<_>>();
        let done_queues = (0..settings.threads)
            .map(|_| std::sync::mpsc::channel::<Done>())
            .collect::<Vec<_>>();

        // Extract all of the shared data into Vecs
        let friend_queue =
            task_queues.iter().map(|t| t.stealer()).collect::<Vec<_>>();
        let friend_done =
            done_queues.iter().map(|t| t.0.clone()).collect::<Vec<_>>();

        let mut workers = task_queues
            .into_iter()
            .zip(done_queues.into_iter().map(|t| t.1))
            .enumerate()
            .map(|(thread_index, (queue, done))| Worker {
                thread_index,
                octree: if thread_index == 0 {
                    Octree::new()
                } else {
                    Octree::empty()
                },
                queue,
                done,
                friend_queue: friend_queue.clone(),
                friend_done: friend_done.clone(),
                data: Default::default(),
                storage: Default::default(),
            })
            .collect::<Vec<_>>();

        let root = CellIndex::default();
        let r = workers[0].octree.eval_cell(
            &eval,
            &mut Default::default(),
            &mut Default::default(),
            root,
            settings,
        );
        let c = match r {
            CellResult::Full => Cell::Full,
            CellResult::Empty => Cell::Empty,
            CellResult::Leaf(leaf) => Cell::Leaf { leaf, thread: 0 },
            CellResult::Recurse { index, eval } => {
                // Inject the recursive task into worker[0]'s queue
                workers[0].queue.push(Task {
                    eval,
                    parent: root,
                    index,
                });
                Cell::Branch { index, thread: 0 }
            }
        };
        workers[0].octree.record(0, c.into());

        if matches!(c, Cell::Branch { .. }) {
            let threads = std::sync::RwLock::new(vec![
                std::thread::current();
                settings.threads as usize
            ]);
            let counter = &AtomicUsize::new(0);
            let out: Vec<Octree> = std::thread::scope(|s| {
                let mut handles = vec![];
                for w in workers {
                    let thread_ref = &threads;
                    handles.push(
                        s.spawn(move || w.run(thread_ref, counter, settings)),
                    );
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            Octree::merge(&out)
        } else {
            workers.into_iter().next().unwrap().octree
        }
    }

    /// Runs a single worker to completion as part of a worker group
    pub fn run(
        mut self,
        threads: &std::sync::RwLock<Vec<std::thread::Thread>>,
        counter: &AtomicUsize,
        settings: Settings,
    ) -> Octree {
        ////////////////////////////////////////////////////////////////////////
        // Setup: build the `threads` array for later waking.
        //
        // Record our current index
        let mut w = threads.write().unwrap();
        let thread_count = w.len();
        w[self.thread_index] = std::thread::current();
        counter.fetch_add(1, Ordering::Release);

        // Wake all of the other workers; if everyone has registered themselves,
        // then the counter will be at thread_count and everyone will continue.
        for (i, t) in w.iter().enumerate() {
            if i != self.thread_index {
                t.unpark();
            }
        }
        drop(w);

        // Wait until every thread has installed itself into the array
        while counter.load(Ordering::Acquire) < thread_count {
            std::thread::park();
        }

        // At this point, every thread can borrow this array immutably
        let threads = threads.read().unwrap();
        let threads = threads.as_slice();

        ////////////////////////////////////////////////////////////////////////

        loop {
            // First, check to see if anyone has finished a task and sent us
            // back the result.  Otherwise, keep going.
            match self.done.try_recv() {
                Ok(v) => {
                    self.octree.record(v.cell_index, v.child);
                    continue;
                }
                Err(TryRecvError::Disconnected) => panic!(),
                Err(TryRecvError::Empty) => {
                    // nothing to do here
                }
            }

            let t = self.queue.pop().map(|t| (t, self.thread_index)).or_else(
                || {
                    use crossbeam_deque::Steal;
                    // Try stealing from all of our friends (but not ourselves)
                    for i in 1..self.friend_queue.len() {
                        let i =
                            (i + self.thread_index) % self.friend_queue.len();
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
                },
            );

            if let Some((mut task, source)) = t {
                // Each task represents 8 cells, so evaluate them one by one
                // here and return results.
                let mut any_recurse = false;
                for i in Corner::iter() {
                    let sub_cell = task.parent.child(task.index, i);

                    let r = match self.octree.eval_cell(
                        &task.eval,
                        &mut self.data,
                        &mut self.storage,
                        sub_cell,
                        settings,
                    ) {
                        CellResult::Empty => Cell::Empty,
                        CellResult::Full => Cell::Full,
                        CellResult::Leaf(leaf) => Cell::Leaf {
                            leaf,
                            thread: self.thread_index as u8,
                        },
                        CellResult::Recurse { index, eval } => {
                            self.queue.push(Task {
                                eval,
                                parent: sub_cell,
                                index,
                            });
                            any_recurse = true;
                            Cell::Branch {
                                index,
                                thread: self.thread_index as u8,
                            }
                        }
                    };

                    if source != self.thread_index {
                        // Send the result back on the wire
                        self.friend_done[source]
                            .send(Done {
                                cell_index: sub_cell.index,
                                child: r.into(),
                            })
                            .unwrap();
                    } else {
                        // Store the result locally
                        self.octree.record(sub_cell.index, r.into());
                    }
                }
                // If we pushed anything to our queue, then let other threads
                // wake up to try stealing tasks.
                if any_recurse {
                    for (i, t) in threads.iter().enumerate() {
                        if i != self.thread_index {
                            t.unpark();
                        }
                    }
                }

                if let Some(e) = Arc::get_mut(&mut task.eval) {
                    self.storage
                        .interval_storage
                        .extend(e.interval.take().and_then(|s| s.take()));
                    self.storage
                        .float_storage
                        .extend(e.float_slice.take().and_then(|s| s.take()));
                    self.storage
                        .grad_storage
                        .extend(e.grad_slice.take().and_then(|s| s.take()));
                }

                // We've successfully done some work, so start the loop again
                // from the top and see what else needs to be done.
                continue;
            }

            // At this point, the thread doesn't have any work to do, so we'll
            // consider putting it to sleep.  However, if every other thread is
            // sleeping, then we're ready to exit; we'll wake them all up.
            let c = 1 + (counter.fetch_add(256, Ordering::Release) >> 8);
            if c == thread_count {
                // Wake up the other threads, so they notice that we're done
                for (i, t) in threads.iter().enumerate() {
                    if i != self.thread_index {
                        t.unpark();
                    }
                }
                break;
            }
            // There are other active threads, so park ourselves and wait for
            // someone else to wake us up.
            std::thread::park();
            if counter.load(Ordering::Acquire) >> 8 == thread_count {
                break;
            }
            // Back to the grind
            counter.fetch_sub(256, Ordering::Release);
        }

        // Cleanup, flushing the done queue
        loop {
            match self.done.try_recv() {
                Ok(v) => self.octree.record(v.cell_index, v.child),
                Err(TryRecvError::Disconnected) => panic!(),
                Err(TryRecvError::Empty) => break,
            }
        }
        self.octree
    }
}
