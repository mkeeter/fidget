//! Multithreaded octree construction
use super::pool::{QueuePool, ThreadContext, ThreadPool};
use crate::{
    eval::Family,
    mesh::{
        cell::{Cell, CellData, CellIndex},
        octree::{
            BranchResult, CellResult, EvalData, EvalGroup, EvalStorage,
            OctreeBuilder,
        },
        types::Corner,
        Octree, Settings,
    },
};
use std::sync::{mpsc::TryRecvError, Arc};

/// Represents a chunk of work that should be handled by a worker
///
/// Specifically, the work is to subdivide the given cell and evaluate its 8
/// octants, sending results back to the parent (which is numbered implicitly
/// based on what queue we stole this from).
#[derive(Clone)]
struct Task<I: Family> {
    data: Arc<TaskData<I>>,
}

impl<I: Family> std::ops::Deref for Task<I> {
    type Target = TaskData<I>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<I: Family> Task<I> {
    /// Builds a new root task
    ///
    /// The root task is from worker 0 with the default cell index
    fn new(eval: Arc<EvalGroup<I>>) -> Self {
        Self {
            data: Arc::new(TaskData {
                eval,
                target_cell: CellIndex::default(),
                assigned_by: 0,
                parent: None,
            }),
        }
    }

    fn child(
        &self,
        eval: Arc<EvalGroup<I>>,
        target_cell: CellIndex,
        assigned_by: usize,
    ) -> Self {
        Self {
            data: Arc::new(TaskData {
                eval,
                target_cell,
                assigned_by,
                parent: Some(self.data.clone()),
            }),
        }
    }

    fn release(self, storage: &mut EvalStorage<I>) {
        if let Ok(mut t) = Arc::try_unwrap(self.data) {
            loop {
                if let Ok(e) = Arc::try_unwrap(t.eval) {
                    storage.claim(e);
                }
                if let Some(parent) =
                    t.parent.and_then(|n| Arc::try_unwrap(n).ok())
                {
                    t = parent;
                } else {
                    break;
                }
            }
        }
    }
}

struct TaskData<I: Family> {
    eval: Arc<EvalGroup<I>>,

    /// Thread in which the parent cell lives
    assigned_by: usize,

    /// Parent cell, which must be an `Invalid` cell waiting for population
    target_cell: CellIndex,

    parent: Option<Arc<TaskData<I>>>,
}

struct Done<I: Family> {
    /// The task that we have finished evaluating
    task: Task<I>,

    /// The resulting cell
    ///
    /// - Filled and empty cells are owned (trivially) by the receiving octree
    /// - Leafs must be converted into vertex and hermite data and owned by the
    ///   receiving octree.
    /// - Branches point to the sending octree, which may be in another worker
    result: BranchResult,

    /// Thread index of the worker that did this work
    completed_by: usize,
}

pub struct OctreeWorker<I: Family> {
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
    octree: OctreeBuilder,

    /// Incoming completed tasks from other threads
    done: std::sync::mpsc::Receiver<Done<I>>,

    /// Our queue of tasks
    queue: QueuePool<Task<I>>,

    /// When a worker finishes a task, it returns it through these queues
    ///
    /// Like `friend_queue`, there's one per thread, including the worker's own
    /// thread; it would be silly to send stuff back to your own thread via the
    /// queue (rather than storing it directly).
    friend_done: Vec<std::sync::mpsc::Sender<Done<I>>>,

    /// Per-thread local data for evaluation, to avoid allocation churn
    data: EvalData<I>,
}

impl<I: Family> OctreeWorker<I> {
    pub fn scheduler(eval: Arc<EvalGroup<I>>, settings: Settings) -> Octree {
        let task_queues = QueuePool::new(settings.threads as usize);
        let done_queues = std::iter::repeat_with(std::sync::mpsc::channel)
            .take(settings.threads as usize)
            .collect::<Vec<_>>();
        let friend_done =
            done_queues.iter().map(|t| t.0.clone()).collect::<Vec<_>>();

        let mut workers = task_queues
            .into_iter()
            .zip(done_queues.into_iter().map(|t| t.1))
            .enumerate()
            .map(|(thread_index, (queue, done))| OctreeWorker {
                thread_index,
                octree: if thread_index == 0 {
                    OctreeBuilder::new()
                } else {
                    OctreeBuilder::empty()
                },
                queue,
                done,
                friend_done: friend_done.clone(),
                data: Default::default(),
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
            CellResult::Done(cell) => Some(cell),
            CellResult::Recurse(eval) => {
                // Inject the recursive task into worker[0]'s queue
                workers[0].queue.push(Task::new(eval));
                None
            }
        };
        if let Some(c) = c {
            workers[0].octree.record(0, c.into());
            workers.into_iter().next().unwrap().octree.into()
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
        let mut ctx = threads.start(self.thread_index);
        let mut storage = Default::default();
        loop {
            // First, check to see if anyone has finished a task and sent us
            // back the result.  Otherwise, keep going.
            match self.done.try_recv() {
                Ok(v) => {
                    ctx.popped();
                    self.on_done(
                        v.result,
                        &v.task.data,
                        v.completed_by,
                        &mut ctx,
                    );
                    continue;
                }
                Err(TryRecvError::Disconnected) => panic!(),
                Err(TryRecvError::Empty) => {
                    // nothing to do here
                }
            }

            if let Some(task) = self.queue.pop() {
                // Each task represents 8 cells, so evaluate them one by one
                // here and return results.

                // Prepare a set of 8x cells for storage
                let index = self.octree.o.cells.len();
                for _ in Corner::iter() {
                    self.octree.o.cells.push(Cell::Invalid.into());
                }

                for i in Corner::iter() {
                    let sub_cell = task.target_cell.child(index, i);

                    match self.octree.eval_cell(
                        &task.eval,
                        &mut self.data,
                        &mut storage,
                        sub_cell,
                        settings,
                    ) {
                        // If this child is finished, then record it locally.
                        // If it's a branching cell, then we'll let a caller
                        // fill it in eventually (via the done queue).
                        CellResult::Done(cell) => self.record(
                            sub_cell.index,
                            cell.into(),
                            &task.data,
                            &mut ctx,
                        ),
                        CellResult::Recurse(eval) => {
                            self.queue.push(task.child(
                                eval,
                                sub_cell,
                                self.thread_index,
                            ));
                        }
                    };
                }

                // If we pushed anything to our queue, then let other threads
                // wake up to try stealing tasks.
                if self.queue.changed() {
                    ctx.wake();
                } else {
                    // Try to recycle tape storage!
                    //
                    // We may or may not have unique ownership of the task; we
                    // didn't push anything to the queue, but may have cloned
                    // the task when sending a Done message back to the caller.
                    task.release(&mut storage);
                }

                // We've successfully done some work, so start the loop again
                // from the top and see what else needs to be done.
                continue;
            }

            if !ctx.sleep() {
                break;
            }
        }

        // At this point, the `done` queue should be flushed
        assert_eq!(self.done.try_recv().err(), Some(TryRecvError::Empty));

        self.octree.into()
    }

    fn on_done(
        &mut self,
        result: BranchResult,
        task: &Arc<TaskData<I>>,
        completed_by: usize,
        ctx: &mut ThreadContext,
    ) {
        let r = match result {
            BranchResult::Empty => Cell::Empty,
            BranchResult::Full => Cell::Full,
            BranchResult::Branch(index) => Cell::Branch {
                index,
                thread: completed_by as u8,
            },
            BranchResult::Leaf(pos, hermite) => {
                self.octree.record_leaf(pos, hermite)
            }
        };
        if let Some(parent) = task.parent.as_ref() {
            // Store the result locally, recursing up
            self.record(task.target_cell.index, r.into(), parent, ctx);
        } else {
            // Store the result locally, but don't recurse (because this is a
            // root task and has nowhere to go)
            self.octree.record(task.target_cell.index, r.into());
        }
    }

    /// Records the cell at the given index and recurses upwards
    ///
    /// This cell must be one of the 8 children of the given task.
    ///
    /// If complete, the results are sent upstream to the giver of the task;
    /// this may be the same worker thread or another worker.
    fn record(
        &mut self,
        index: usize,
        cell: CellData,
        parent_task: &Arc<TaskData<I>>,
        ctx: &mut ThreadContext,
    ) {
        self.octree.record(index, cell);

        // Check to see whether this is the last cell in the cluster of 8
        let target_cell = parent_task.target_cell;
        let Some(r) = self.octree.check_done(target_cell, index & !7)
            else { return };

        // It's safe to unwrap `task` here because the only task lacking a
        // parent is at the root of the tree, which is never a set of 8
        // children; `check_done` will cause us to bail out early.
        if parent_task.assigned_by == self.thread_index {
            self.on_done(r, parent_task, self.thread_index, ctx);
        } else {
            // Send the result back on the wire
            ctx.pushed();
            self.friend_done[parent_task.assigned_by]
                .send(Done {
                    task: Task {
                        data: parent_task.clone(),
                    },
                    result: r,
                    completed_by: self.thread_index,
                })
                .unwrap();
            ctx.wake_one(parent_task.assigned_by);
        }
    }
}
