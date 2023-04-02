use std::sync::{mpsc::TryRecvError, Arc};

use super::{
    cell::{Cell, CellData, CellIndex},
    octree::{CellResult, EvalData, EvalGroup, EvalStorage},
    pool::ThreadPool,
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

    /// Parent cell, which must be an `Invalid` cell waiting for population
    parent: CellIndex,
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
    /// Global index of this worker thread
    ///
    /// For example, this is the thread's own index in `friend_queue` and
    /// `friend_done`.
    thread_index: usize,

    /// The under-construction octree.
    ///
    /// This octree may not be complete; worker 0 is guaranteed to contain the
    /// root, and other works may contain fragmentary branches that point to
    /// each other in a tree structure.
    octree: Octree,

    /// Our personal queue of tasks to complete
    ///
    /// Other threads may steal from this queue!
    queue: crossbeam_deque::Worker<Task<I>>,

    /// Incoming completed tasks from other threads
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
            CellResult::Full => Some(Cell::Full),
            CellResult::Empty => Some(Cell::Empty),
            CellResult::Leaf(leaf) => Some(Cell::Leaf { leaf, thread: 0 }),
            CellResult::Recurse(eval) => {
                // Inject the recursive task into worker[0]'s queue
                workers[0].queue.push(Task { eval, parent: root });
                None
            }
        };
        if let Some(c) = c {
            workers[0].octree.record(0, c.into());
            workers.into_iter().next().unwrap().octree
        } else {
            let pool = &ThreadPool::new(settings.threads as usize);
            let out: Vec<Octree> = std::thread::scope(|s| {
                let mut handles = vec![];
                for w in workers {
                    handles.push(s.spawn(move || w.run(pool, settings)));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            Octree::merge(&out)
        }
    }

    /// Runs a single worker to completion as part of a worker group
    pub fn run(mut self, threads: &ThreadPool, settings: Settings) -> Octree {
        let ctx = threads.start(self.thread_index);

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

            if let Some((task, source)) = t {
                // Each task represents 8 cells, so evaluate them one by one
                // here and return results.
                let mut any_recurse = false;

                // Prepare a set of 8x cells for storage
                let index = self.octree.cells.len();
                for _ in Corner::iter() {
                    self.octree.cells.push(Cell::Invalid.into());
                }

                for i in Corner::iter() {
                    let sub_cell = task.parent.child(index, i);

                    let r = match self.octree.eval_cell(
                        &task.eval,
                        &mut self.data,
                        &mut self.storage,
                        sub_cell,
                        settings,
                    ) {
                        CellResult::Empty => Some(Cell::Empty),
                        CellResult::Full => Some(Cell::Full),
                        CellResult::Leaf(leaf) => Some(Cell::Leaf {
                            leaf,
                            thread: self.thread_index as u8,
                        }),
                        CellResult::Recurse(eval) => {
                            self.queue.push(Task {
                                eval,
                                parent: sub_cell,
                            });
                            any_recurse = true;
                            None
                        }
                    };
                    // If this child is finished, then record it locally.  If
                    // it's a branching cell, then we'll let a caller fill it in
                    // eventually (via the done queue).
                    if let Some(r) = r {
                        self.octree.record(sub_cell.index, r.into());
                    }
                }

                // We're done evaluating the task; record that it is a branch
                // pointing to this thread's data array.
                let r = Cell::Branch {
                    index,
                    thread: self.thread_index as u8,
                };
                if source != self.thread_index {
                    // Send the result back on the wire
                    self.friend_done[source]
                        .send(Done {
                            cell_index: task.parent.index,
                            child: r.into(),
                        })
                        .unwrap();
                } else {
                    // Store the result locally
                    self.octree.record(task.parent.index, r.into());
                }

                // If we pushed anything to our queue, then let other threads
                // wake up to try stealing tasks.
                if any_recurse {
                    ctx.wake();
                }

                // Try to recycle tape storage
                if let Ok(e) = Arc::try_unwrap(task.eval) {
                    self.storage.claim(e);
                }

                // We've successfully done some work, so start the loop again
                // from the top and see what else needs to be done.
                continue;
            }

            if !ctx.sleep() {
                break;
            }
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
