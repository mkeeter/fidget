//! Dual contouring implementation

use super::{
    cell::{Cell, CellIndex},
    frame::{Frame, XYZ, YZX, ZXY},
    pool::ThreadPool,
    types::{Corner, Edge, X, Y, Z},
    Mesh, Octree,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub enum Task {
    Cell(CellIndex),
    FaceXYZ(CellIndex, CellIndex),
    FaceYZX(CellIndex, CellIndex),
    FaceZXY(CellIndex, CellIndex),
    EdgeXYZ(CellIndex, CellIndex, CellIndex, CellIndex),
    EdgeYZX(CellIndex, CellIndex, CellIndex, CellIndex),
    EdgeZXY(CellIndex, CellIndex, CellIndex, CellIndex),
}

pub trait FrameToTask {
    fn face(a: CellIndex, b: CellIndex) -> Task;
    fn edge(a: CellIndex, b: CellIndex, c: CellIndex, d: CellIndex) -> Task;
}

impl FrameToTask for XYZ {
    fn face(a: CellIndex, b: CellIndex) -> Task {
        Task::FaceXYZ(a, b)
    }
    fn edge(a: CellIndex, b: CellIndex, c: CellIndex, d: CellIndex) -> Task {
        Task::EdgeXYZ(a, b, c, d)
    }
}

impl FrameToTask for YZX {
    fn face(a: CellIndex, b: CellIndex) -> Task {
        Task::FaceYZX(a, b)
    }
    fn edge(a: CellIndex, b: CellIndex, c: CellIndex, d: CellIndex) -> Task {
        Task::EdgeYZX(a, b, c, d)
    }
}

impl FrameToTask for ZXY {
    fn face(a: CellIndex, b: CellIndex) -> Task {
        Task::FaceZXY(a, b)
    }
    fn edge(a: CellIndex, b: CellIndex, c: CellIndex, d: CellIndex) -> Task {
        Task::EdgeZXY(a, b, c, d)
    }
}

/// Multithreaded worker for mesh generation
pub struct DcWorker<'a> {
    /// Global index of this worker thread
    ///
    /// For example, this is the thread's own index in `friend_queue`.
    ///
    /// By construction, this must always fit into a `u8`, but it's stored as a
    /// `usize` for convenience.
    thread_index: usize,

    /// Target octree
    octree: &'a Octree,

    /// Map from octree vertex index (position in [`Octree::verts`]) to mesh
    /// vertex.
    ///
    /// This starts as all 0; individual threads claim vertices by doing an
    /// atomic compare-exchange.  A claimed vertex is divided into bits as
    /// follows:
    /// - The top bit is always 1 (so 0 is not a valid claimed index)
    /// - The next 8 bits indicate which thread has claimed the vertex
    /// - The remaining bits are an index into that thread's [`DcWorker::verts`]
    map: &'a [AtomicUsize],
    tris: Vec<nalgebra::Vector3<usize>>,
    verts: Vec<nalgebra::Vector3<f32>>,

    /// Our personal queue of tasks to complete
    ///
    /// Other threads may steal from this queue!
    queue: crossbeam_deque::Worker<Task>,

    /// Queues from which we can steal other workers' tasks
    ///
    /// Each task has `thread_count` friend queues, including its own; it would
    /// be silly to steal from your own queue, but that keeps the code cleaner.
    friend_queue: Vec<crossbeam_deque::Stealer<Task>>,
}

// TODO: lots of this duplicates stuff in `octree.rs`, and the multithreaded
// worker architecture mimicks that in `worker.rs`
impl<'a> DcWorker<'a> {
    pub fn scheduler(octree: &Octree, threads: u8) -> Mesh {
        let task_queues = (0..threads)
            .map(|_| crossbeam_deque::Worker::<Task>::new_lifo())
            .collect::<Vec<_>>();

        let friend_queue =
            task_queues.iter().map(|t| t.stealer()).collect::<Vec<_>>();

        let map = octree
            .verts
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        let workers = task_queues
            .into_iter()
            .enumerate()
            .map(|(thread_index, queue)| DcWorker {
                thread_index,
                octree,
                map: &map,
                queue,
                friend_queue: friend_queue.clone(),
                tris: vec![],
                verts: vec![],
            })
            .collect::<Vec<_>>();
        workers[0].queue.push(Task::Cell(CellIndex::default()));

        let pool = &ThreadPool::new(threads as usize);
        let out: Vec<_> = std::thread::scope(|s| {
            let mut handles = vec![];
            for w in workers {
                handles.push(s.spawn(move || w.run(pool)));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Calculate offsets within the global merged mesh
        let mut vert_offsets = vec![0];

        for (_, verts) in &out {
            let i = vert_offsets.last().unwrap();
            vert_offsets.push(i + verts.len());
        }
        let tri_count = out.iter().map(|(t, _)| t.len()).sum();

        // We'll be building a single mesh as output, but the mesh will be
        // constructed in parallel with individual threads copying data into
        // chunks of the output triangle and vertex arrays.
        let mut mesh = Mesh {
            vertices: vec![
                nalgebra::Vector3::zeros();
                *vert_offsets.last().unwrap()
            ],
            triangles: vec![nalgebra::Vector3::zeros(); tri_count],
        };

        let mut slice = mesh.vertices.as_mut_slice();
        let mut out_verts = vec![];
        for n in out.iter().map(|(_, v)| v.len()) {
            let (a, b) = slice.split_at_mut(n);
            out_verts.push(a);
            slice = b;
        }

        let mut slice = mesh.triangles.as_mut_slice();
        let mut out_tris = vec![];
        for n in out.iter().map(|(t, _)| t.len()) {
            let (a, b) = slice.split_at_mut(n);
            out_tris.push(a);
            slice = b;
        }

        // Multi-thread copying!
        let vert_offsets_ref = &vert_offsets;
        std::thread::scope(|s| {
            for ((tris, verts), (out_t, out_v)) in out
                .into_iter()
                .zip(out_tris.into_iter().zip(out_verts.into_iter()))
            {
                s.spawn(move || {
                    out_t
                        .iter_mut()
                        .zip(tris.iter().map(|t| {
                            t.map(|v| {
                                let thread = (v >> 55) & 0xFF;
                                let i = v & ((1 << 55) - 1);
                                vert_offsets_ref[thread] + i
                            })
                        }))
                        .for_each(|(o, i)| *o = i);
                    out_v
                        .iter_mut()
                        .zip(verts.into_iter())
                        .for_each(|(o, i)| *o = i);
                });
            }
        });

        mesh
    }

    pub fn run(
        mut self,
        pool: &ThreadPool,
    ) -> (Vec<nalgebra::Vector3<usize>>, Vec<nalgebra::Vector3<f32>>) {
        let ctx = pool.start(self.thread_index);

        loop {
            let t = self.queue.pop().or_else(|| {
                use crossbeam_deque::Steal;
                // Try stealing from all of our friends (but not ourselves)
                for i in 1..self.friend_queue.len() {
                    let i = (i + self.thread_index) % self.friend_queue.len();
                    let q = &self.friend_queue[i];
                    loop {
                        match q.steal() {
                            Steal::Success(v) => return Some(v),
                            Steal::Empty => break,
                            Steal::Retry => continue,
                        }
                    }
                }
                None
            });

            if let Some(task) = t {
                // Evaluate a single task, pushing any recursive tasks to our
                // own queue.
                let any_recurse = match task {
                    Task::Cell(i) => dc_cell(&mut self, i),
                    Task::FaceXYZ(a, b) => dc_face::<_, XYZ>(&mut self, a, b),
                    Task::FaceYZX(a, b) => dc_face::<_, YZX>(&mut self, a, b),
                    Task::FaceZXY(a, b) => dc_face::<_, ZXY>(&mut self, a, b),
                    Task::EdgeXYZ(a, b, c, d) => {
                        dc_edge::<_, XYZ>(&mut self, a, b, c, d)
                    }
                    Task::EdgeYZX(a, b, c, d) => {
                        dc_edge::<_, YZX>(&mut self, a, b, c, d)
                    }
                    Task::EdgeZXY(a, b, c, d) => {
                        dc_edge::<_, ZXY>(&mut self, a, b, c, d)
                    }
                };

                // Wake other threads, since there could be work available
                if any_recurse {
                    ctx.wake();
                }

                // We've successfully done some work, so start the loop again
                // from the top and see what else needs to be done.
                continue;
            }

            if !ctx.sleep() {
                break;
            }
        }

        (self.tris, self.verts)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait DcCore {
    /// Looks up the given octree vertex
    fn get_vertex(&mut self, i: usize, cell: CellIndex) -> usize;

    /// Pushes a task for evaluation
    fn push(&mut self, task: Task);

    fn octree(&self) -> &Octree;
    fn triangle(&mut self, a: usize, b: usize, c: usize);
}

impl DcCore for DcWorker<'_> {
    fn push(&mut self, task: Task) {
        self.queue.push(task);
    }

    /// Looks up the given octree vertex
    ///
    /// This function attempts to claim it for this thread, but if that fails,
    /// then it returns the other thread's claimed vertex index (which includes
    /// the thread ID in the upper bits).
    fn get_vertex(&mut self, i: usize, cell: CellIndex) -> usize {
        // Build our thread + vertex index
        let mut next = self.verts.len();
        assert!(next < (1 << 55));
        next |= 1 << 63;
        next |= (self.thread_index as usize) << 55;

        match self.map[i].compare_exchange(
            0,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.verts.push(cell.pos(self.octree.verts[i]));
                next
            }
            Err(i) => i,
        }
    }

    fn octree(&self) -> &Octree {
        self.octree
    }

    fn triangle(&mut self, a: usize, b: usize, c: usize) {
        self.tris.push(nalgebra::Vector3::new(a, b, c))
    }
}

#[allow(clippy::modulo_one, clippy::identity_op, unused_parens)]
pub fn dc_cell<C: DcCore>(core: &mut C, cell: CellIndex) -> bool {
    if let Cell::Branch { index, .. } = core.octree().cells[cell.index].into() {
        debug_assert_eq!(index % 8, 0);
        for i in Corner::iter() {
            core.push(Task::Cell(core.octree().child(cell, i)));
        }

        dc_faces::<C, XYZ>(core, cell);
        dc_faces::<C, YZX>(core, cell);
        dc_faces::<C, ZXY>(core, cell);

        for i in [false, true] {
            core.push(Task::EdgeXYZ(
                core.octree().child(cell, (X * i)),
                core.octree().child(cell, (X * i) | Y),
                core.octree().child(cell, (X * i) | Y | Z),
                core.octree().child(cell, (X * i) | Z),
            ));
            core.push(Task::EdgeYZX(
                core.octree().child(cell, (Y * i)),
                core.octree().child(cell, (Y * i) | Z),
                core.octree().child(cell, (Y * i) | X | Z),
                core.octree().child(cell, (Y * i) | X),
            ));
            core.push(Task::EdgeZXY(
                core.octree().child(cell, (Z * i)),
                core.octree().child(cell, (Z * i) | X),
                core.octree().child(cell, (Z * i) | X | Y),
                core.octree().child(cell, (Z * i) | Y),
            ));
        }
        true
    } else {
        false
    }
}

/// Queues up `Face` tasks on all four face adjacencies in the given frame
fn dc_faces<C: DcCore, T: Frame + FrameToTask>(core: &mut C, cell: CellIndex) {
    let (t, u, v) = T::frame();
    for c in [Corner::new(0), u.into(), v.into(), u | v] {
        core.push(T::face(
            core.octree().child(cell, c),
            core.octree().child(cell, c | t),
        ));
    }
}

/// Handles two cells which share a common face
///
/// `lo` is below `hi` on the `T` axis; the cells share a `UV` face where
/// `T-U-V` is a right-handed coordinate system.
#[allow(unused_parens)]
pub fn dc_face<C: DcCore, T: Frame + FrameToTask>(
    core: &mut C,
    lo: CellIndex,
    hi: CellIndex,
) -> bool
where
    <T as Frame>::Next: FrameToTask,
    <<T as Frame>::Next as Frame>::Next: FrameToTask,
{
    if core.octree().is_leaf(lo) && core.octree().is_leaf(hi) {
        return false;
    }
    let (t, u, v) = T::frame();
    core.push(T::face(
        core.octree().child(lo, t),
        core.octree().child(hi, Corner::new(0)),
    ));
    core.push(T::face(
        core.octree().child(lo, t | u),
        core.octree().child(hi, u),
    ));
    core.push(T::face(
        core.octree().child(lo, t | v),
        core.octree().child(hi, v),
    ));
    core.push(T::face(
        core.octree().child(lo, t | u | v),
        core.octree().child(hi, u | v),
    ));
    for i in [false, true] {
        core.push(<T::Next as FrameToTask>::edge(
            core.octree().child(lo, (u * i) | t),
            core.octree().child(lo, (u * i) | v | t),
            core.octree().child(hi, (u * i) | v),
            core.octree().child(hi, (u * i)),
        ));
        core.push(<<T::Next as Frame>::Next as FrameToTask>::edge(
            core.octree().child(lo, (v * i) | t),
            core.octree().child(hi, (v * i)),
            core.octree().child(hi, (v * i) | u),
            core.octree().child(lo, (v * i) | u | t),
        ));
    }
    true
}

/// Handles four cells that share a common edge aligned on axis `T`
///
/// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
/// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
///
/// - `dc_edge<X>` is `[0, Y, Y | Z, Z]`
/// - `dc_edge<Y>` is `[0, Z, Z | X, X]`
/// - `dc_edge<Z>` is `[0, X, X | Y, Y]`
#[allow(clippy::identity_op, unused_parens)]
pub fn dc_edge<C: DcCore, T: Frame + FrameToTask>(
    core: &mut C,
    a: CellIndex,
    b: CellIndex,
    c: CellIndex,
    d: CellIndex,
) -> bool {
    let cs = [a, b, c, d];
    if cs.iter().all(|v| core.octree().is_leaf(*v)) {
        // If any of the leafs are Empty or Full, then this edge can't
        // include a sign change.  TODO: can we make this any -> all if we
        // collapse empty / filled leafs into Empty / Full cells?
        let leafs =
            cs.map(|cell| match core.octree().cells[cell.index].into() {
                Cell::Leaf { leaf, .. } => Some(leaf),
                Cell::Empty | Cell::Full => None,
                Cell::Branch { .. } => unreachable!(),
                Cell::Invalid => panic!(),
            });
        if leafs.iter().any(Option::is_none) {
            return false;
        }
        let leafs = leafs.map(Option::unwrap);

        // TODO: check for a sign change on this edge
        let (t, u, v) = T::frame();
        let sign_change_count = leafs
            .iter()
            .zip([u | v, v.into(), Corner::new(0), u.into()])
            .filter(|(leaf, c)| {
                (leaf.mask & (1 << c.index()) == 0)
                    != (leaf.mask & (1 << (*c | t).index()) == 0)
            })
            .count();
        if sign_change_count == 0 {
            return false;
        }
        debug_assert_eq!(sign_change_count, 4);

        let verts = [
            leafs[0].edge(Edge::new((t.index() * 4 + 3) as u8)),
            leafs[1].edge(Edge::new((t.index() * 4 + 2) as u8)),
            leafs[2].edge(Edge::new((t.index() * 4 + 0) as u8)),
            leafs[3].edge(Edge::new((t.index() * 4 + 1) as u8)),
        ];

        // Pick the intersection vertex based on the deepest cell
        let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();
        let i = core.get_vertex(
            leafs[deepest].index + verts[deepest].edge.0 as usize,
            cs[deepest],
        );
        // Helper function to extract other vertices
        let mut vert = |i: usize| {
            core.get_vertex(leafs[i].index + verts[i].vert.0 as usize, cs[i])
        };
        let vs = [vert(0), vert(1), vert(2), vert(3)];

        // Pick a triangle winding depending on the edge direction
        let winding = if leafs[0].mask & (1 << (u | v).index()) == 0 {
            3
        } else {
            1
        };
        for j in 0..4 {
            core.triangle(vs[j], vs[(j + winding) % 4], i);
        }
        false
    } else {
        let (t, u, v) = T::frame();
        for i in [false, true] {
            core.push(T::edge(
                core.octree().child(a, (t * i) | u | v),
                core.octree().child(b, (t * i) | v),
                core.octree().child(c, (t * i)),
                core.octree().child(d, (t * i) | u),
            ));
        }
        true
    }
}
