//! Multithreaded dual contouring
use super::pool::{QueuePool, ThreadPool};
use crate::mesh::{
    cell::{CellIndex, CellVertex},
    dc::{dc_cell, dc_edge, dc_face, DcBuilder},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{X, Y, Z},
    Mesh, Octree,
};
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU64, Ordering},
};

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
    map: &'a [AtomicU64],
    tris: Vec<nalgebra::Vector3<u64>>,
    verts: Vec<nalgebra::Vector3<f32>>,

    /// Our personal queue of tasks to complete, along with references to other
    /// queues within the pool (for stealing)
    queue: QueuePool<Task>,
}

impl<'a> DcWorker<'a> {
    pub fn scheduler(octree: &Octree, threads: NonZeroUsize) -> Mesh {
        let queues = QueuePool::new(threads);

        let map = octree
            .verts
            .iter()
            .map(|_| AtomicU64::new(0))
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

        let pool = &ThreadPool::new(threads);
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
                                let thread = ((v >> 55) & 0xFF) as usize;
                                let i: usize =
                                    (v & ((1 << 55) - 1)).try_into().unwrap();
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
    ) -> (Vec<nalgebra::Vector3<u64>>, Vec<nalgebra::Vector3<f32>>) {
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
    // We need to reserve 1 byte for multithreading info, so we'll force the use
    // of a u64 here (rather than a usize, which is 32-bit on some platforms)
    type VertexIndex = u64;

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

    fn triangle(&mut self, a: u64, b: u64, c: u64) {
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
        _cell: CellIndex,
        verts: &[CellVertex],
    ) -> u64 {
        // Build our thread + vertex index
        let mut next = self.verts.len() as u64;
        assert!(next < (1 << 55));
        next |= 1 << 63;
        next |= (self.thread_index as u64) << 55;

        match self.map[i].compare_exchange(
            0,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.verts.push(verts[i].pos);
                next
            }
            Err(i) => i,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
