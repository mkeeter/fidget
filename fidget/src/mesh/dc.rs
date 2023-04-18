//! Dual contouring implementation

use super::{
    cell::{Cell, CellIndex, CellVertex},
    frame::{Frame, XYZ, YZX, ZXY},
    pool::{QueuePool, ThreadPool},
    types::{Corner, Edge, X, Y, Z},
    Mesh, Octree,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
enum Task {
    Cell(CellIndex),
    FaceXYZ(CellIndex, CellIndex),
    FaceYZX(CellIndex, CellIndex),
    FaceZXY(CellIndex, CellIndex),
    EdgeXYZ(CellIndex, CellIndex, CellIndex, CellIndex),
    EdgeYZX(CellIndex, CellIndex, CellIndex, CellIndex),
    EdgeZXY(CellIndex, CellIndex, CellIndex, CellIndex),
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

    /// Our personal queue of tasks to complete, along with references to other
    /// queues within the pool (for stealing)
    queue: QueuePool<Task>,
}

impl<'a> DcWorker<'a> {
    pub fn scheduler(octree: &Octree, threads: u8) -> Mesh {
        let queues = QueuePool::new(threads as usize);

        let map = octree
            .verts
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>();

        let mut workers = queues
            .into_iter()
            .enumerate()
            .map(|(thread_index, queue)| DcWorker {
                thread_index,
                octree,
                map: &map,
                queue,
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
        let mut ctx = pool.start(self.thread_index);

        loop {
            if let Some(task) = self.queue.pop() {
                // Each task represents 8 cells, so evaluate them one by one
                // here and return results.
                match task {
                    Task::Cell(i) => dc_cell(self.octree, i, &mut self),
                    Task::FaceXYZ(a, b) => {
                        dc_face::<XYZ, _>(self.octree, a, b, &mut self)
                    }
                    Task::FaceYZX(a, b) => {
                        dc_face::<YZX, _>(self.octree, a, b, &mut self)
                    }
                    Task::FaceZXY(a, b) => {
                        dc_face::<ZXY, _>(self.octree, a, b, &mut self)
                    }
                    Task::EdgeXYZ(a, b, c, d) => {
                        dc_edge::<XYZ, _>(self.octree, a, b, c, d, &mut self)
                    }
                    Task::EdgeYZX(a, b, c, d) => {
                        dc_edge::<YZX, _>(self.octree, a, b, c, d, &mut self)
                    }
                    Task::EdgeZXY(a, b, c, d) => {
                        dc_edge::<ZXY, _>(self.octree, a, b, c, d, &mut self)
                    }
                };

                // Wake other threads, since there could be work available
                if self.queue.changed() {
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

#[allow(clippy::modulo_one, clippy::identity_op, unused_parens)]
impl DcBuilder for DcWorker<'_> {
    fn cell(&mut self, _octree: &Octree, cell: CellIndex) {
        self.queue.push(Task::Cell(cell));
    }

    fn face<F: Frame>(&mut self, _octree: &Octree, a: CellIndex, b: CellIndex) {
        match F::frame().0 {
            X => self.queue.push(Task::FaceXYZ(a, b)),
            Y => self.queue.push(Task::FaceYZX(a, b)),
            Z => self.queue.push(Task::FaceZXY(a, b)),
            _ => unreachable!(),
        }
    }

    fn edge<F: Frame>(
        &mut self,
        _octree: &Octree,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
    ) {
        match F::frame().0 {
            X => self.queue.push(Task::EdgeXYZ(a, b, c, d)),
            Y => self.queue.push(Task::EdgeYZX(a, b, c, d)),
            Z => self.queue.push(Task::EdgeZXY(a, b, c, d)),
            _ => unreachable!(),
        }
    }

    fn triangle(&mut self, a: usize, b: usize, c: usize) {
        self.tris.push(nalgebra::Vector3::new(a, b, c))
    }

    /// Looks up the given octree vertex
    ///
    /// This function attempts to claim it for this thread, but if that fails,
    /// then it returns the other thread's claimed vertex index (which includes
    /// the thread ID in the upper bits).
    fn vertex(
        &mut self,
        i: usize,
        cell: CellIndex,
        verts: &[CellVertex],
    ) -> usize {
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
                self.verts.push(cell.pos(verts[i]));
                next
            }
            Err(i) => i,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait DcBuilder {
    fn cell(&mut self, octree: &Octree, cell: CellIndex);
    fn face<F: Frame>(&mut self, octree: &Octree, a: CellIndex, b: CellIndex);
    fn edge<F: Frame>(
        &mut self,
        octree: &Octree,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
    );

    /// Callback for an invalid leaf vertex
    ///
    /// This occurs if a non-manifold (multi-vertex) cell is a **larger**
    /// neighbor of a small cell during meshing, because we can't decide which
    /// of the vertices to choose.  It indicates that the larger cell should be
    /// subdivided.
    fn invalid_leaf_vert(&mut self, a: CellIndex) {
        panic!("invalid leaf vertex at {a:?}");
    }

    /// Callback when a triangle fan is finished
    ///
    /// The default implementation does nothing; `DcFixup` uses this to forget
    /// the most recent batch of vertices, since it only needs them for local
    /// checking.
    fn fan_done(&mut self) {
        // Nothing to do here
    }

    /// Record the given triangle
    ///
    /// Vertices are indices given by calls to [`Self::vertex`]
    ///
    /// The vertices are given in a clockwise winding with the intersection
    /// vertex (i.e. the one on the edge) always last.
    fn triangle(&mut self, a: usize, b: usize, c: usize);

    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`](super::Octree::verts).
    fn vertex(
        &mut self,
        v: usize,
        cell: CellIndex,
        verts: &[CellVertex],
    ) -> usize;
}

pub fn dc_cell<B: DcBuilder>(octree: &Octree, cell: CellIndex, out: &mut B) {
    if let Cell::Branch { index, .. } = octree.cells[cell.index].into() {
        debug_assert_eq!(index % 8, 0);
        for i in Corner::iter() {
            out.cell(octree, octree.child(cell, i));
        }

        // Helper function for DC face calls
        fn dc_faces<T: Frame, B: DcBuilder>(
            octree: &Octree,
            cell: CellIndex,
            out: &mut B,
        ) {
            let (t, u, v) = T::frame();
            for c in [Corner::new(0), u.into(), v.into(), u | v] {
                out.face::<T>(
                    octree,
                    octree.child(cell, c),
                    octree.child(cell, c | t),
                );
            }
        }
        dc_faces::<XYZ, _>(octree, cell, out);
        dc_faces::<YZX, _>(octree, cell, out);
        dc_faces::<ZXY, _>(octree, cell, out);

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<XYZ>(
                octree,
                octree.child(cell, (X * i)),
                octree.child(cell, (X * i) | Y),
                octree.child(cell, (X * i) | Y | Z),
                octree.child(cell, (X * i) | Z),
            );
            out.edge::<YZX>(
                octree,
                octree.child(cell, (Y * i)),
                octree.child(cell, (Y * i) | Z),
                octree.child(cell, (Y * i) | X | Z),
                octree.child(cell, (Y * i) | X),
            );
            out.edge::<ZXY>(
                octree,
                octree.child(cell, (Z * i)),
                octree.child(cell, (Z * i) | X),
                octree.child(cell, (Z * i) | X | Y),
                octree.child(cell, (Z * i) | Y),
            );
        }
    }
}

/// Handles two cells which share a common face
///
/// `lo` is below `hi` on the `T` axis; the cells share a `UV` face where
/// `T-U-V` is a right-handed coordinate system.
pub fn dc_face<T: Frame, B: DcBuilder>(
    octree: &Octree,
    lo: CellIndex,
    hi: CellIndex,
    out: &mut B,
) {
    if octree.is_leaf(lo) && octree.is_leaf(hi) {
        return;
    }
    let (t, u, v) = T::frame();
    out.face::<T>(
        octree,
        octree.child(lo, t),
        octree.child(hi, Corner::new(0)),
    );
    out.face::<T>(octree, octree.child(lo, t | u), octree.child(hi, u));
    out.face::<T>(octree, octree.child(lo, t | v), octree.child(hi, v));
    out.face::<T>(octree, octree.child(lo, t | u | v), octree.child(hi, u | v));
    #[allow(unused_parens)]
    for i in [false, true] {
        out.edge::<T::Next>(
            octree,
            octree.child(lo, (u * i) | t),
            octree.child(lo, (u * i) | v | t),
            octree.child(hi, (u * i) | v),
            octree.child(hi, (u * i)),
        );
        out.edge::<<T::Next as Frame>::Next>(
            octree,
            octree.child(lo, (v * i) | t),
            octree.child(hi, (v * i)),
            octree.child(hi, (v * i) | u),
            octree.child(lo, (v * i) | u | t),
        );
    }
}

/// Handles four cells that share a common edge aligned on axis `T`
///
/// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
/// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
///
/// - `dc_edge<X>` is `[0, Y, Y | Z, Z]`
/// - `dc_edge<Y>` is `[0, Z, Z | X, X]`
/// - `dc_edge<Z>` is `[0, X, X | Y, Y]`
pub fn dc_edge<T: Frame, B: DcBuilder>(
    octree: &Octree,
    a: CellIndex,
    b: CellIndex,
    c: CellIndex,
    d: CellIndex,
    out: &mut B,
) {
    let cs = [a, b, c, d];
    if cs.iter().all(|v| octree.is_leaf(*v)) {
        // If any of the leafs are Empty or Full, then this edge can't
        // include a sign change.  TODO: can we make this any -> all if we
        // collapse empty / filled leafs into Empty / Full cells?
        let leafs = cs.map(|cell| match octree.cells[cell.index].into() {
            Cell::Leaf(leaf) => Some(leaf),
            Cell::Empty | Cell::Full => None,
            Cell::Branch { .. } => unreachable!(),
            Cell::Invalid => panic!(),
        });
        if leafs.iter().any(Option::is_none) {
            return;
        }
        let leafs = leafs.map(Option::unwrap);

        // TODO: should we pick a canonically deepest leaf instead of the first
        // among the four that's at the deepest depth?
        let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();

        let (t, _u, _v) = T::frame();

        // Each leaf has an edge associated with it
        #[allow(clippy::identity_op)]
        let edges = [
            Edge::new((t.index() * 4 + 3) as u8),
            Edge::new((t.index() * 4 + 2) as u8),
            Edge::new((t.index() * 4 + 0) as u8),
            Edge::new((t.index() * 4 + 1) as u8),
        ];

        // Find the starting sign of the relevant edge, bailing out early if
        // there isn't a sign change here.  All of the deepest edges should show
        // the same sign change, so it doesn't matter which one we pick here.
        let starting_sign = {
            let (start, end) = edges[deepest].corners();
            let start = leafs[deepest].mask & (1 << start.index()) == 0;
            let end = leafs[deepest].mask & (1 << end.index()) == 0;
            // If there is no sign change, then there's nothing to do here.
            if start == end {
                return;
            }
            start
        };

        // Iterate over each of the edges, assigning a vertex if the sign change
        // lines up.
        let mut failed = false;
        let mut verts = [None; 4];
        for i in 0..4 {
            if cs[i].depth == cs[deepest].depth {
                let (start, end) = edges[i].corners();
                let s = leafs[i].mask & (1 << start.index()) == 0;
                let e = leafs[i].mask & (1 << end.index()) == 0;
                debug_assert_eq!(s, starting_sign);
                debug_assert_eq!(e, !starting_sign);
                verts[i] = leafs[i].edge(edges[i]);
            } else {
                // We declare that only *manifold* leaf cells can be neighbors
                // to smaller leaf cells.  This means that there's only one
                // vertex to pick here.
                let mut iter =
                    (0..12).filter_map(|j| leafs[i].edge(Edge::new(j)));
                verts[i] = iter.next();
                if iter.any(|other| other.vert != verts[i].unwrap().vert) {
                    // This panics in normal meshing, and records the vertex
                    // otherwise.
                    out.invalid_leaf_vert(cs[i]);

                    // Accumulate a flag so that we have time to check every
                    // leaf, but we're going to return early.
                    failed = true;
                }
            }
        }

        // If any of the leafs are invalid due to their neighbors, then return
        // immediately; we'll retry once they have been subdivided.
        if failed {
            return;
        }

        let verts = verts.map(Option::unwrap);

        // Pick the intersection vertex based on the deepest cell
        let i = out.vertex(
            leafs[deepest].index + verts[deepest].edge.0 as usize,
            cs[deepest],
            &octree.verts,
        );
        // Helper function to extract other vertices
        let mut vert = |i: usize| {
            out.vertex(
                leafs[i].index + verts[i].vert.0 as usize,
                cs[i],
                &octree.verts,
            )
        };
        let vs = [vert(0), vert(1), vert(2), vert(3)];

        // Pick a triangle winding depending on the edge direction
        //
        // As always, we have to sample the deepest leaf's edge to be sure that
        // we get the correct value.
        let winding = if starting_sign { 3 } else { 1 };
        for j in 0..4 {
            if cs[j].index != cs[(j + winding) % 4].index {
                out.triangle(vs[j], vs[(j + winding) % 4], i)
            }
        }

        // Note that we have completed a triangle fan.  This is used by the
        // DcFixup to forget its triangles, since it doesn't need to preserve
        // them on a long-term basis.
        out.fan_done();
    } else {
        let (t, u, v) = T::frame();

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<T>(
                octree,
                octree.child(a, (t * i) | u | v),
                octree.child(b, (t * i) | v),
                octree.child(c, (t * i)),
                octree.child(d, (t * i) | u),
            )
        }
    }
}
