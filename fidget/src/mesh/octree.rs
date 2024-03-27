//! An octree data structure and implementation of Manifold Dual Contouring

use super::{
    builder::MeshBuilder,
    cell::{Cell, CellData, CellIndex, CellVertex, Leaf},
    dc::DcBuilder,
    fixup::DcFixup,
    frame::Frame,
    gen::CELL_TO_VERT_TO_EDGES,
    mt::{DcWorker, OctreeWorker},
    qef::QuadraticErrorSolver,
    types::{Axis, Corner, Edge, EdgeMask, Face, FaceMask},
    Mesh, Settings,
};
use crate::eval::{BulkEvaluator, Shape, Tape, TracingEvaluator};
use std::{num::NonZeroUsize, sync::Arc, sync::OnceLock};

/// Helper struct to contain a set of matched evaluators
///
/// Note that this is `Send + Sync` and can be used with shared references!
pub struct EvalGroup<S: Shape> {
    pub shape: S,

    // TODO: passing around an `Arc<EvalGroup>` ends up with two layers of
    // indirection (since the tapes also contain `Arc`); could we flatten
    // them out?  (same with the shape, which is usually an `Arc`)
    pub interval: OnceLock<<S::IntervalEval as TracingEvaluator>::Tape>,
    pub float_slice: OnceLock<<S::FloatSliceEval as BulkEvaluator>::Tape>,
    pub grad_slice: OnceLock<<S::GradSliceEval as BulkEvaluator>::Tape>,
}

impl<S: Shape> EvalGroup<S> {
    fn new(shape: S) -> Self {
        Self {
            shape,
            interval: OnceLock::new(),
            float_slice: OnceLock::new(),
            grad_slice: OnceLock::new(),
        }
    }
    fn interval_tape(
        &self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::IntervalEval as TracingEvaluator>::Tape {
        self.interval.get_or_init(|| {
            self.shape.interval_tape(storage.pop().unwrap_or_default())
        })
    }
    fn float_slice_tape(
        &self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::FloatSliceEval as BulkEvaluator>::Tape {
        self.float_slice.get_or_init(|| {
            self.shape
                .float_slice_tape(storage.pop().unwrap_or_default())
        })
    }
    fn grad_slice_tape(
        &self,
        storage: &mut Vec<S::TapeStorage>,
    ) -> &<S::GradSliceEval as BulkEvaluator>::Tape {
        self.grad_slice.get_or_init(|| {
            self.shape
                .grad_slice_tape(storage.pop().unwrap_or_default())
        })
    }
}

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    /// The top two bits determine cell type
    pub(crate) cells: Vec<CellData>,

    /// Cell vertices, given as positions within the cell
    ///
    /// The `bool` in the tuple indicates whether the vertex was clamped to fit
    /// into the cell's bounding box.
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    pub(crate) verts: Vec<CellVertex>,
}

impl Octree {
    /// Merges a set of octrees constructed across multiple workers
    ///
    /// # Panics
    /// All cross-octree references must be valid
    pub(crate) fn merge(os: &[Octree]) -> Octree {
        // Calculate offsets within the global merged Octree
        let mut cell_offsets = vec![0];
        let mut vert_offsets = vec![0];
        for o in os {
            let i = cell_offsets.last().unwrap();
            cell_offsets.push(i + o.cells.len());

            let i = vert_offsets.last().unwrap();
            vert_offsets.push(i + o.verts.len());
        }

        let mut out = Octree {
            cells: Vec::with_capacity(*cell_offsets.last().unwrap()),
            verts: Vec::with_capacity(*vert_offsets.last().unwrap()),
        };

        for (t, o) in os.iter().enumerate() {
            for c in &o.cells {
                let c: Cell = match (*c).into() {
                    c @ (Cell::Empty | Cell::Full | Cell::Invalid) => c,
                    Cell::Branch { index, thread } => Cell::Branch {
                        index: cell_offsets[thread as usize] + index,
                        thread: 0,
                    },
                    Cell::Leaf(Leaf { mask, index }) => Cell::Leaf(Leaf {
                        index: vert_offsets[t] + index,
                        mask,
                    }),
                };
                out.cells.push(c.into());
            }
            out.verts.extend(o.verts.iter().cloned());
        }
        out
    }

    /// Builds an octree to the given depth
    ///
    /// The shape is evaluated on the region `[-1, 1]` on all axes
    pub fn build<S: Shape + Clone>(shape: &S, settings: Settings) -> Self {
        // Transform the shape given our bounds
        let t = settings.bounds.transform();
        if t == nalgebra::Transform::identity() {
            Self::build_inner(shape, settings)
        } else {
            let shape = shape.clone().apply_transform(t.into());
            let mut out = Self::build_inner(&shape, settings);

            // Apply the reverse transform to vertex coordinates
            let ti = t.matrix().try_inverse().unwrap();
            for v in &mut out.verts {
                let p: nalgebra::Point3<f32> = v.pos.into();
                let q = ti.transform_point(&p);
                v.pos = q.coords;
            }
            out
        }
    }

    fn build_inner<S: Shape + Clone>(shape: &S, settings: Settings) -> Self {
        let eval = Arc::new(EvalGroup::new(shape.clone()));

        let mut octree = if settings.threads == 0 {
            let mut out = OctreeBuilder::new();
            out.recurse(&eval, CellIndex::default(), settings);
            out.into()
        } else {
            OctreeWorker::scheduler(eval.clone(), settings)
        };

        // If we can't refine any further, then return right away
        if settings.min_depth == settings.max_depth {
            return octree;
        }

        loop {
            let mut fixup = DcFixup::new(octree.cells.len(), &settings);
            fixup.cell(&octree, CellIndex::default());
            let num_fix = fixup.needs_fixing.iter().filter(|i| **i).count();
            if num_fix == 0 {
                break;
            }
            // Translate from an Octree back to an OctreeBuilder; specifically,
            // the index field in a Cell::Leaf points into the `leafs` array,
            // rather than the `verts` array.
            let mut cells = vec![];
            let mut leafs = vec![];
            for c in octree.cells {
                cells.push(if let Cell::Leaf(Leaf { mask, index }) = c.into() {
                    let leaf_index = leafs.len();
                    leafs.push(LeafData {
                        vert_index: index,
                        hermite_index: None,
                    });
                    Cell::Leaf(Leaf {
                        mask,
                        index: leaf_index,
                    })
                    .into()
                } else {
                    c
                })
            }
            let mut b = OctreeBuilder {
                o: Octree {
                    cells,
                    verts: octree.verts,
                },
                leafs,
                hermite: vec![LeafHermiteData::default()],
                hermite_slots: vec![],
                eval_float_slice: S::new_float_slice_eval(),
                eval_grad_slice: S::new_grad_slice_eval(),
                eval_interval: S::new_interval_eval(),
                tape_storage: vec![],
                shape_storage: vec![],
                workspace: Default::default(),
            };
            b.refine(&eval, CellIndex::default(), &fixup.needs_fixing);
            octree = b.into();
        }
        octree
    }

    /// Recursively walks the dual of the octree, building a mesh
    pub fn walk_dual(&self, settings: Settings) -> Mesh {
        let mut mesh = MeshBuilder::default();

        if settings.threads == 0 {
            mesh.cell(self, CellIndex::default());
            mesh.take()
        } else {
            DcWorker::scheduler(self, settings.threads)
        }
    }

    pub(crate) fn is_leaf(&self, cell: CellIndex) -> bool {
        match self[cell].into() {
            Cell::Leaf(..) | Cell::Full | Cell::Empty => true,
            Cell::Branch { .. } => false,
            Cell::Invalid => panic!(),
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    pub(crate) fn child<C: Into<Corner>>(
        &self,
        cell: CellIndex,
        child: C,
    ) -> CellIndex {
        let child = child.into();

        match self[cell].into() {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index, .. } => cell.child(index, child),
            Cell::Invalid => panic!(),
        }
    }

    pub(crate) fn edge_mask(
        &self,
        cell: CellIndex,
        edge: Edge,
    ) -> Option<EdgeMask> {
        let (a, b) = edge.corners();
        match self[cell].into() {
            Cell::Empty => Some(EdgeMask::new(0b00)),
            Cell::Full => Some(EdgeMask::new(0b11)),
            Cell::Leaf(Leaf { mask, .. }) => {
                let lo = mask & (1 << a.index()) != 0;
                let hi = mask & (1 << b.index()) != 0;
                Some(EdgeMask::new(lo as u8 + ((hi as u8) << 1)))
            }
            Cell::Branch { index, .. } => {
                let Some(lo) = self.edge_mask(cell.child(index, a), edge)
                else {
                    return None;
                };
                let Some(hi) = self.edge_mask(cell.child(index, b), edge)
                else {
                    return None;
                };
                let center = lo.0 & (0b10) != 0;
                if center == (lo.0 & 0b01 != 0)
                    || center == (hi.0 & (0b10) != 0)
                {
                    Some(EdgeMask::new((lo.0 & 0b01) | (hi.0 & 0b10)))
                } else {
                    None
                }
            }
            Cell::Invalid => panic!(),
        }
    }

    pub(crate) fn face_mask(
        &self,
        cell: CellIndex,
        face: Face,
    ) -> Option<FaceMask> {
        let t = face.axis();
        let u = t.next();
        let v = u.next();
        let f = if face.sign() {
            t.into()
        } else {
            Corner::new(0)
        };
        let corners = [f, f | u, f | v, f | u | v];
        match self[cell].into() {
            Cell::Empty => Some(FaceMask::new(0b0000)),
            Cell::Full => Some(FaceMask::new(0b1111)),
            Cell::Leaf(Leaf { mask, .. }) => {
                let mut out = 0;
                for (i, c) in corners.iter().enumerate() {
                    if mask & (1 << c.index()) != 0 {
                        out |= 1 << i;
                    }
                }
                Some(FaceMask::new(out))
            }
            Cell::Branch { index, .. } => {
                let masks =
                    corners.map(|c| self.face_mask(cell.child(index, c), face));
                if masks.iter().any(Option::is_none) {
                    return None;
                }
                let masks = masks.map(Option::unwrap);

                let center = masks[0].0 & (1 << 3) != 0;
                let corners = [
                    masks[0].0 & (1 << 0) != 0,
                    masks[1].0 & (1 << 1) != 0,
                    masks[2].0 & (1 << 2) != 0,
                    masks[3].0 & (1 << 3) != 0,
                ];
                // The center must match at least one of the corners; otherwise,
                // this is non-manifold.
                if corners.iter().all(|corner| center != *corner) {
                    return None;
                }
                // Each edge's center value must match at least one of the
                // connected corners; otherwise, this is non-manifold
                let u_edge_lo = masks[0].0 & (1 << 1) != 0;
                if u_edge_lo != corners[0] && u_edge_lo != corners[1] {
                    return None;
                }
                let u_edge_hi = masks[3].0 & (1 << 2) != 0;
                if u_edge_hi != corners[2] && u_edge_hi != corners[3] {
                    return None;
                }
                let v_edge_lo = masks[0].0 & (1 << 2) != 0;
                if v_edge_lo != corners[0] && v_edge_lo != corners[2] {
                    return None;
                }
                let v_edge_hi = masks[3].0 & (1 << 1) != 0;
                if v_edge_hi != corners[1] && v_edge_hi != corners[3] {
                    return None;
                }

                Some(FaceMask::new(
                    corners[0] as u8
                        | (corners[1] as u8) << 1
                        | (corners[2] as u8) << 2
                        | (corners[3] as u8) << 3,
                ))
            }
            Cell::Invalid => panic!(),
        }
    }
}

impl std::ops::Index<CellIndex> for Octree {
    type Output = CellData;

    fn index(&self, i: CellIndex) -> &Self::Output {
        &self.cells[i.index]
    }
}

impl std::ops::IndexMut<CellIndex> for Octree {
    fn index_mut(&mut self, i: CellIndex) -> &mut Self::Output {
        &mut self.cells[i.index]
    }
}

/// Data structure for an under-construction octree
#[derive(Debug)]
pub(crate) struct OctreeBuilder<S: Shape> {
    /// Internal octree
    ///
    /// Note that in this internal octree, the `index` field of leaf nodes
    /// points into `LeafData`, rather than into `verts` directly.  This is
    /// necessary because each leaf represents _two_ indices: one into `verts`
    /// and one (optional) index into `hermite`.
    pub(crate) o: Octree,

    /// Leaf data contains indexes into `verts` and `hermite`
    leafs: Vec<LeafData>,

    /// Hermite cell data for active leafs
    ///
    /// This should be kept small; we don't need to save it for leafs that are
    /// no longer active (i.e. no longer in contention for merging)
    ///
    /// Slot 0 is reserved and should always be populated with a dummy value, so
    /// that we can use an `Option<NonZeroUsize>` elsewhere.
    hermite: Vec<LeafHermiteData>,

    /// Available slots in the `hermite` array
    hermite_slots: Vec<usize>,

    eval_float_slice: S::FloatSliceEval,
    eval_interval: S::IntervalEval,
    eval_grad_slice: S::GradSliceEval,

    pub tape_storage: Vec<S::TapeStorage>,
    pub shape_storage: Vec<S::Storage>,
    workspace: S::Workspace,
}

impl<S: Shape> Default for OctreeBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Shape> From<OctreeBuilder<S>> for Octree {
    fn from(o: OctreeBuilder<S>) -> Self {
        // Convert from "leaf index into self.leafs" (in the builder) to
        // "leaf index into self.verts" (in the resulting Octree)
        let cells =
            o.o.cells
                .into_iter()
                .map(|c| {
                    if let Cell::Leaf(Leaf { mask, index }) = c.into() {
                        Cell::Leaf(Leaf {
                            mask,
                            index: o.leafs[index].vert_index,
                        })
                        .into()
                    } else {
                        c
                    }
                })
                .collect();
        Self {
            cells,
            verts: o.o.verts,
        }
    }
}

impl<S: Shape> OctreeBuilder<S> {
    /// Builds a new octree, which allocates data for 8 root cells
    pub(crate) fn new() -> Self {
        Self {
            o: Octree {
                cells: vec![Cell::Invalid.into(); 8],
                verts: vec![],
            },
            leafs: vec![],
            hermite: vec![LeafHermiteData::default()],
            hermite_slots: vec![],
            eval_float_slice: S::new_float_slice_eval(),
            eval_grad_slice: S::new_grad_slice_eval(),
            eval_interval: S::new_interval_eval(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    /// Builds a new empty octree
    ///
    /// This still allocates data to reserve the lowest slot in `hermite`
    pub(crate) fn empty() -> Self {
        Self {
            o: Octree {
                cells: vec![],
                verts: vec![],
            },
            leafs: vec![],
            hermite: vec![LeafHermiteData::default()],
            hermite_slots: vec![],

            eval_float_slice: S::new_float_slice_eval(),
            eval_grad_slice: S::new_grad_slice_eval(),
            eval_interval: S::new_interval_eval(),

            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    /// Stores hermite data in the local array
    fn push_hermite(&mut self, d: LeafHermiteData) -> usize {
        if let Some(s) = self.hermite_slots.pop() {
            self.hermite[s] = d;
            s
        } else {
            let s = self.hermite.len();
            self.hermite.push(d);
            s
        }
    }

    /// Releases hermite data from the local array
    fn pop_hermite(&mut self, i: usize) {
        self.hermite_slots.push(i);
    }

    /// Records the given cell into the provided index
    ///
    /// The index must be valid already; this does not modify the cells vector.
    ///
    /// # Panics
    /// If the index exceeds the bounds of the cell vector, or the cell is
    /// already populated.
    pub(crate) fn record(&mut self, index: usize, cell: CellData) {
        debug_assert_eq!(self.o.cells[index], Cell::Invalid.into());
        self.o.cells[index] = cell;
    }

    /// Evaluates a single cell in the octree
    ///
    /// Leaf data is stored in `self.verts`; cell results are **not** written
    /// back to the `cells` array, because the cell may be rooted in a different
    /// octree (e.g. on another thread).
    pub(crate) fn eval_cell(
        &mut self,
        eval: &Arc<EvalGroup<S>>,
        cell: CellIndex,
        settings: Settings,
    ) -> CellResult<S> {
        let (i, r) = self
            .eval_interval
            .eval(
                eval.interval_tape(&mut self.tape_storage),
                cell.bounds.x,
                cell.bounds.y,
                cell.bounds.z,
                &[],
            )
            .unwrap();
        if i.upper() < 0.0 {
            CellResult::Done(Cell::Full)
        } else if i.lower() > 0.0 {
            CellResult::Done(Cell::Empty)
        } else {
            let sub_tape = if S::simplify_tree_during_meshing(cell.depth) {
                let s = self.shape_storage.pop().unwrap_or_default();
                r.map(|r| {
                    Arc::new(EvalGroup::new(
                        eval.shape.simplify(r, s, &mut self.workspace).unwrap(),
                    ))
                })
            } else {
                None
            };
            if cell.depth == settings.min_depth as usize {
                let eval = sub_tape.unwrap_or_else(|| eval.clone());
                let out = CellResult::Done(self.leaf(&eval, cell));
                if let Ok(t) = Arc::try_unwrap(eval) {
                    self.reclaim(t);
                }
                out
            } else {
                CellResult::Recurse(sub_tape.unwrap_or_else(|| eval.clone()))
            }
        }
    }

    /// Records the vertex and hermite data for the given leaf
    ///
    /// Does not record the leaf cell itself; it's returned for the caller to
    /// record.
    pub(crate) fn record_leaf(
        &mut self,
        pos: CellVertex,
        hermite: LeafHermiteData,
    ) -> Cell {
        let hermite_index = self.push_hermite(hermite);

        let vert_index = self.o.verts.len();
        self.o.verts.push(pos);

        // Install cell intersections, of which there must only
        // be one (since we only collapse manifold cells)
        let edges = &CELL_TO_VERT_TO_EDGES[hermite.mask as usize];
        debug_assert_eq!(edges.len(), 1);
        for e in edges[0] {
            let i = hermite.intersections[e.to_undirected().index()];
            self.o.verts.push(CellVertex { pos: i.pos.xyz() });
        }

        let leaf_index = self.leafs.len();
        self.leafs.push(LeafData {
            vert_index,
            hermite_index: NonZeroUsize::new(hermite_index),
        });

        Cell::Leaf(Leaf {
            mask: hermite.mask,
            index: leaf_index,
        })
    }

    /// Recurse down the octree, building the given cell
    fn recurse(
        &mut self,
        eval: &Arc<EvalGroup<S>>,
        cell: CellIndex,
        settings: Settings,
    ) {
        match self.eval_cell(eval, cell, settings) {
            CellResult::Done(c) => self.o[cell] = c.into(),
            CellResult::Recurse(sub_eval) => {
                let index = self.o.cells.len();
                for _ in Corner::iter() {
                    self.o.cells.push(Cell::Invalid.into());
                }
                for i in Corner::iter() {
                    let cell = cell.child(index, i);
                    self.recurse(&sub_eval, cell, settings);
                }

                if let Ok(t) = Arc::try_unwrap(sub_eval) {
                    self.reclaim(t);
                }

                let r = self.check_done(cell, index).unwrap();

                self.o[cell] = match r {
                    BranchResult::Empty => Cell::Empty,
                    BranchResult::Full => Cell::Full,
                    BranchResult::Branch(index) => {
                        Cell::Branch { index, thread: 0 }
                    }
                    BranchResult::Leaf(pos, hermite) => {
                        self.record_leaf(pos, hermite)
                    }
                }
                .into();
            }
        }
    }

    /// Evaluates the given leaf
    ///
    /// Writes the leaf vertex to `self.o.verts`, hermite data to
    /// `self.hermite`, and the leaf data to `self.leafs`.  Does **not** write
    /// anything to `self.o.cells`; the cell is returned instead.
    fn leaf(&mut self, eval: &EvalGroup<S>, cell: CellIndex) -> Cell {
        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::iter() {
            let (x, y, z) = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        let out = self
            .eval_float_slice
            .eval(
                eval.float_slice_tape(&mut self.tape_storage),
                &xs,
                &ys,
                &zs,
                &[],
            )
            .unwrap();
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, &v)| v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

        // Early exit if the cell is completely empty or full
        if mask == 0 {
            return Cell::Empty;
        } else if mask == 255 {
            return Cell::Full;
        }

        // Start and endpoints in 3D space for intersection searches
        let mut start = [nalgebra::Vector3::zeros(); 12];
        let mut end = [nalgebra::Vector3::zeros(); 12];
        let mut edge_count = 0;

        // Convert from the corner mask to start and end-points in 3D space,
        // relative to the cell bounds (i.e. as a `Matrix3<u16>`).
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            for e in *vs {
                // Find the axis that's being used by this edge
                let axis = e.start().index() ^ e.end().index();
                debug_assert_eq!(axis.count_ones(), 1);
                debug_assert!(axis < 8);

                // Pick a position closer to the filled side
                let (a, b) = if e.end().index() & axis != 0 {
                    (0, u16::MAX)
                } else {
                    (u16::MAX, 0)
                };

                // Convert the intersection to a 3D position
                let mut v = nalgebra::Vector3::zeros();
                let i = (axis.trailing_zeros() + 1) % 3;
                let j = (axis.trailing_zeros() + 2) % 3;
                v[i as usize] = if e.start() & Axis::new(1 << i) {
                    u16::MAX
                } else {
                    0
                };
                v[j as usize] = if e.start() & Axis::new(1 << j) {
                    u16::MAX
                } else {
                    0
                };

                v[axis.trailing_zeros() as usize] = a;
                start[edge_count] = v;
                v[axis.trailing_zeros() as usize] = b;
                end[edge_count] = v;
                edge_count += 1;
            }
        }
        // Slice off the unused sections of the arrays
        let start = &mut start[..edge_count]; // always inside
        let end = &mut end[..edge_count]; // always outside

        // Scratch arrays for edge search
        const EDGE_SEARCH_SIZE: usize = 16;
        const EDGE_SEARCH_DEPTH: usize = 4;
        let xs =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let ys =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let zs =
            &mut [0f32; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];

        // This part looks hairy, but it's just doing an N-ary search along each
        // edge to find the intersection point.
        for _ in 0..EDGE_SEARCH_DEPTH {
            // Populate edge search arrays
            let mut i = 0;
            for (start, end) in start.iter().zip(end.iter()) {
                for j in 0..EDGE_SEARCH_SIZE {
                    let pos = ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - j - 1) as u32)
                        + (end.map(|i| i as u32) * j as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32);
                    debug_assert!(pos.max() <= u16::MAX.into());

                    let pos = cell.pos(pos.map(|p| p as u16));
                    xs[i] = pos.x;
                    ys[i] = pos.y;
                    zs[i] = pos.z;
                    i += 1;
                }
            }
            debug_assert_eq!(i, EDGE_SEARCH_SIZE * edge_count);

            // Do the actual evaluation
            let out = self
                .eval_float_slice
                .eval(
                    eval.float_slice_tape(&mut self.tape_storage),
                    xs,
                    ys,
                    zs,
                    &[],
                )
                .unwrap();

            // Update start and end positions based on evaluation
            for ((start, end), search) in start
                .iter_mut()
                .zip(end.iter_mut())
                .zip(out.chunks(EDGE_SEARCH_SIZE))
            {
                // The search must be inside-to-outside
                debug_assert!(search[0] < 0.0);
                debug_assert!(search[EDGE_SEARCH_SIZE - 1] >= 0.0);
                let frac = search
                    .iter()
                    .enumerate()
                    .find(|(_i, v)| **v >= 0.0)
                    .unwrap()
                    .0;
                debug_assert!(frac > 0);
                debug_assert!(frac < EDGE_SEARCH_SIZE);

                let f = |frac| {
                    ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - frac - 1) as u32)
                        + (end.map(|i| i as u32) * frac as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32)
                };

                let a = f(frac - 1);
                let b = f(frac);

                debug_assert!(a.max() <= u16::MAX.into());
                debug_assert!(b.max() <= u16::MAX.into());
                *start = a.map(|v| v as u16);
                *end = b.map(|v| v as u16);
            }
        }

        // Populate intersections to the average of start and end
        let intersections: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            start
                .iter()
                .zip(end.iter())
                .map(|(a, b)| {
                    ((a.map(|v| v as u32) + b.map(|v| v as u32)) / 2)
                        .map(|v| v as u16)
                })
                .collect();

        for (i, xyz) in intersections.iter().enumerate() {
            let pos = cell.pos(*xyz);
            xs[i] = pos.x;
            ys[i] = pos.y;
            zs[i] = pos.z;
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grads = self
            .eval_grad_slice
            .eval(
                eval.grad_slice_tape(&mut self.tape_storage),
                xs,
                ys,
                zs,
                &[],
            )
            .unwrap();

        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        let mut hermite_cell = LeafHermiteData::new();
        hermite_cell.mask = mask;
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            let mut qef = QuadraticErrorSolver::new();
            for e in vs.iter() {
                let pos = nalgebra::Vector3::new(xs[i], ys[i], zs[i]);
                let grad: nalgebra::Vector4<f32> = grads[i].into();

                qef.add_intersection(pos, grad);

                // Record this intersection in the Hermite data for the leaf
                let edge_index = e.to_undirected().index();
                hermite_cell.intersections[edge_index] = LeafIntersection {
                    pos: nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0),
                    grad,
                };

                i += 1;
            }
            let (pos, err) = qef.solve();
            verts.push(pos);

            // We overwrite the error here, because it's only used when
            // collapsing cells, which only occurs if there's a single vertex;
            // last-error-wins works fine in that case.
            hermite_cell.qef_err = err;
        }

        // TODO: use self.record_leaf here?
        let vert_index = self.o.verts.len();
        self.o.verts.extend(verts);
        self.o.verts.extend(
            intersections
                .into_iter()
                .map(|pos| CellVertex { pos: cell.pos(pos) }),
        );

        let hermite_index = self.push_hermite(hermite_cell);
        debug_assert!(hermite_index > 0);

        let leaf_index = self.leafs.len();
        self.leafs.push(LeafData::new(
            vert_index,
            NonZeroUsize::new(hermite_index).unwrap(),
        ));
        Cell::Leaf(Leaf {
            mask,
            index: leaf_index,
        })
    }

    /// Checks the set of 8 children starting at the given index for completion
    ///
    /// Bails out early if any of them are `Invalid` (indicating that they
    /// haven't been fully populated).  If all are empty or full, then
    /// pro-actively collapses the cells (freeing them if they're at the tail
    /// end of the array).
    pub(crate) fn check_done(
        &mut self,
        cell: CellIndex,
        index: usize,
    ) -> Option<BranchResult> {
        assert_eq!(index % 8, 0);
        let mut full_count = 0;
        let mut empty_count = 0;
        let mut has_branch = false;
        let mut hermite_data = [LeafHermiteData::default(); 8];
        for (i, h) in hermite_data.iter_mut().enumerate() {
            match self.o.cells[index + i].into() {
                Cell::Invalid => {
                    return None;
                }
                Cell::Full => {
                    full_count += 1;
                    h.mask = 255;
                }
                Cell::Empty => {
                    empty_count += 1;
                    h.mask = 0;
                }
                Cell::Branch { .. } => has_branch = true,
                Cell::Leaf(Leaf { .. }) => {
                    // Nothing to here because we don't want to take the hermite
                    // index early; we could exit this loop if any cells are
                    // Invalid and don't want to disturb the leaf in that case.
                }
            }
        }

        // If we haven't returned early due to having an invalid cell, then we
        // can proceed with removing hermite data from the leaf cells, since
        // they won't need it anymore (one way or another).
        for (i, h) in hermite_data.iter_mut().enumerate() {
            if let Cell::Leaf(Leaf { index, .. }) =
                self.o.cells[index + i].into()
            {
                let j = self.leafs[index].hermite_index.take().unwrap().get();
                *h = self.hermite[j];
                self.pop_hermite(j);
            }
        }

        let r = if full_count == 8 {
            BranchResult::Full
        } else if empty_count == 8 {
            BranchResult::Empty
        } else if !has_branch && self.collapsible(index) {
            let mut hermite = LeafHermiteData::merge(hermite_data);

            // Empty / full cells should never be produced here.  The only way to
            // get an empty / full cell is for all eight corners to be empty / full;
            // if that was the case, then either:
            //
            // - The interior vertices match, in which case this should have been
            //   collapsed into a single empty / full cell in `check_done`
            // - The interior vertices *do not* match, in which case the cell should
            //   not be marked as collapsible.
            debug_assert!(hermite.mask != 0);
            debug_assert!(hermite.mask != 255);
            let (pos, new_err) = hermite.solve();
            if new_err < hermite.qef_err * 2.0 && cell.bounds.contains(pos) {
                hermite.qef_err = new_err;
                BranchResult::Leaf(pos, hermite)
            } else {
                BranchResult::Branch(index)
            }
        } else {
            BranchResult::Branch(index)
        };

        // If all of the branches are empty or full, then we're going to
        // record an Empty or Full cell in the parent and don't need the 8x
        // children.
        //
        // We can drop them if they happen to be at the tail end of the
        // octree; otherwise, we'll satisfy ourselves with setting them to
        // invalid.
        if matches!(r, BranchResult::Empty | BranchResult::Full) {
            if index == self.o.cells.len() - 8 {
                self.o.cells.resize(index, Cell::Invalid.into());
            } else {
                self.o.cells[index..index + 8].fill(Cell::Invalid.into())
            }
        }

        Some(r)
    }

    /// Checks whether the set of 8 cells beginning at `root` can be collapsed.
    ///
    /// Only topology is checked, based on the three predicates from "Dual
    /// Contouring of Hermite Data" (Ju et al, 2002), §4.1
    ///
    /// # Panics
    /// `root` must be a multiple of 8, because it points at the root of a
    /// cluster of 8 cells.
    pub(crate) fn collapsible(&self, root: usize) -> bool {
        assert_eq!(root % 8, 0);

        // Unpack cells into a friendlier data type
        let cells = {
            let mut cells = [Cell::Invalid; 8];
            for (&c, o) in self.o.cells[root..root + 8].iter().zip(&mut cells) {
                *o = c.into();
            }
            cells
        };

        let mut mask = 0;
        for (i, &c) in cells.iter().enumerate() {
            let b = match c {
                Cell::Leaf(Leaf { mask, .. }) => {
                    if CELL_TO_VERT_TO_EDGES[mask as usize].len() > 1 {
                        return false;
                    }
                    (mask & (1 << i) != 0) as u8
                }
                Cell::Empty => 0,
                Cell::Full => 1,
                Cell::Branch { .. } => return false,
                Cell::Invalid => panic!(),
            };
            mask |= b << i;
        }

        use super::frame::{XYZ, YZX, ZXY};
        for (t, u, v) in [XYZ::frame(), YZX::frame(), ZXY::frame()] {
            //  - The sign in the middle of a coarse edge must agree with the
            //    sign of at least one of the edge’s two endpoints.
            for i in 0..4 {
                let a = (u * ((i & 1) != 0)) | (v * ((i & 2) != 0));
                let b = a | t;
                let center = cells[a.index()].corner(b);

                if [a, b]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return false;
                }
            }

            //  - The sign in the middle of a coarse face must agree with the
            //    sign of at least one of the face’s four corners.
            for i in 0..2 {
                let a: Corner = (t * (i & 1 == 0)).into();
                let b = a | u;
                let c = a | v;
                let d = a | u | v;

                let center = cells[a.index()].corner(d);

                if [a, b, c, d]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return false;
                }
            }
            //  - The sign in the middle of a coarse cube must agree with the
            //    sign of at least one of the cube’s eight corners.
            for _i in 0..1 {
                // Doing this in the t,u,v loop isn't strictly necessary, but it
                // preserves the structure nicely.
                let center = cells[0].corner(t | u | v);
                if (0..8).all(|v| ((mask & (1 << v)) != 0) != center) {
                    return false;
                }
            }
        }

        // The outer cell must not be empty or full at this point; if it was
        // empty or full and the other conditions had been met, then it should
        // have been collapsed already.
        debug_assert_ne!(mask, 255);
        debug_assert_ne!(mask, 0);

        // TODO: this check may not be necessary, because we're doing *manifold*
        // dual contouring; the collapsed cell can have multiple vertices.
        CELL_TO_VERT_TO_EDGES[mask as usize].len() == 1
    }

    /// Recurse down the octree, splitting the given leaf cells
    fn refine(
        &mut self,
        eval: &Arc<EvalGroup<S>>,
        cell: CellIndex,
        needs_fixing: &[bool],
    ) {
        match self.o[cell].into() {
            Cell::Empty | Cell::Full | Cell::Leaf(..)
                if needs_fixing[cell.index] =>
            {
                // We're going to split this cell and refine it
                let index = self.o.cells.len();
                self.o[cell] = Cell::Branch { index, thread: 0 }.into();

                // Evaluate all 8 leafs
                for i in Corner::iter() {
                    let subcell = cell.child(index, i);
                    let leaf = self.leaf(eval, subcell);
                    match leaf {
                        Cell::Leaf(Leaf { index, .. }) => {
                            // Discard hermite data immediately, because we
                            // aren't collapsing leaves here.
                            let hermite_index =
                                self.leafs[index].hermite_index.take().unwrap();
                            self.pop_hermite(hermite_index.get());
                        }
                        Cell::Empty | Cell::Full => (),
                        Cell::Branch { .. } | Cell::Invalid => panic!(),
                    }
                    // We aren't going to collapse the leafs, so just record
                    // them right away.
                    self.o.cells.push(leaf.into());
                }
            }
            Cell::Empty | Cell::Full | Cell::Leaf(..) => {
                assert!(!needs_fixing[cell.index])
            }
            Cell::Branch { index, .. } => {
                assert!(!needs_fixing[cell.index]);
                for i in Corner::iter() {
                    self.refine(eval, cell.child(index, i), needs_fixing)
                }
            }
            Cell::Invalid => panic!(),
        }
    }

    pub(crate) fn reclaim(&mut self, mut e: EvalGroup<S>) {
        if let Some(s) = e.shape.recycle() {
            self.shape_storage.push(s);
        }
        if let Some(i_tape) = e.interval.take() {
            self.tape_storage.push(i_tape.recycle());
        }
        if let Some(f_tape) = e.float_slice.take() {
            self.tape_storage.push(f_tape.recycle());
        }
        if let Some(g_tape) = e.grad_slice.take() {
            self.tape_storage.push(g_tape.recycle());
        }
    }
}

/// Result of a single cell evaluation
pub enum CellResult<S: Shape> {
    Done(Cell),
    Recurse(Arc<EvalGroup<S>>),
}

/// Result of a branch evaluation (8-fold division)
#[allow(clippy::large_enum_variant)]
pub(crate) enum BranchResult {
    /// The branch can be collapsed into an empty cell
    Empty,
    /// The branch can be collapsed into a full cell
    Full,
    /// The branch remains a branch, with the given index
    Branch(usize),

    /// The branch should be collapsed into a leaf
    ///
    /// The leaf is defined by a single vertex position and hermite data for the
    /// leaf, but is not necessarily written into this octree's data arrays;
    /// when doing multithreaded construction, it may be passed back to a parent
    /// thread and saved there instead.
    Leaf(CellVertex, LeafHermiteData),
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
struct LeafData {
    /// Starting index of vertices within `Octree::verts`
    ///
    /// The total vertex count depends on the leaf mask
    vert_index: usize,

    /// Index of hermite data, if this leaf is still active
    hermite_index: Option<NonZeroUsize>,
}

impl LeafData {
    /// Builds a new leaf data object
    ///
    /// When a leaf is first created, it **must** have hermite data associated
    /// with it; the hermite data may be deleted later in
    /// [`OctreeBuilder::check_done`].
    fn new(vert_index: usize, hermite_index: NonZeroUsize) -> Self {
        Self {
            vert_index,
            hermite_index: Some(hermite_index),
        }
    }
}

#[derive(Copy, Clone, Default, Debug)]
struct LeafIntersection {
    /// Intersection position is xyz; w is 1 if the intersection is present
    pos: nalgebra::Vector4<f32>,
    /// Gradient is xyz; w is the distance field value at this point
    grad: nalgebra::Vector4<f32>,
}

impl From<LeafIntersection> for QuadraticErrorSolver {
    fn from(i: LeafIntersection) -> Self {
        let mut qef = QuadraticErrorSolver::default();
        if i.pos.w != 0.0 {
            qef.add_intersection(i.pos.xyz(), i.grad);
        }
        qef
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LeafHermiteData {
    intersections: [LeafIntersection; 12],
    face_qefs: [QuadraticErrorSolver; 6],
    center_qef: QuadraticErrorSolver,
    qef_err: f32,
    mask: u8,
}

impl Default for LeafHermiteData {
    fn default() -> Self {
        Self {
            intersections: Default::default(),
            face_qefs: Default::default(),
            center_qef: Default::default(),
            qef_err: -1.0,
            mask: 0,
        }
    }
}

impl LeafHermiteData {
    fn new() -> Self {
        Self::default()
    }
    /// Merges an octree subdivision of leaf hermite data
    fn merge(leafs: [LeafHermiteData; 8]) -> Self {
        let mut out = Self::default();
        use super::types::{X, Y, Z};

        // Accumulate intersections along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();
            for edge in 0..4 {
                let mut start = Corner::new(0);
                if edge & 1 != 0 {
                    start = start | u;
                }
                if edge & 2 != 0 {
                    start = start | v;
                }
                let end = start | t;

                // Canonical edge as a value in the 0-12 range
                let edge = Edge::new((t.index() * 4 + edge) as u8);

                // One or the other leaf has an intersection
                let a = leafs[start.index()].intersections[edge.index()];
                let b = leafs[end.index()].intersections[edge.index()];
                match (a.pos.w > 0.0, b.pos.w > 0.0) {
                    (true, false) => out.intersections[edge.index()] = a,
                    (false, true) => out.intersections[edge.index()] = b,
                    (false, false) => (),
                    (true, true) => panic!("duplicate intersection"),
                }
            }
        }

        // Accumulate face QEFs along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = t.next();
            for face in 0..2 {
                let a = if face == 1 { t.into() } else { Corner::new(0) };
                let b = a | u;
                let c = a | v;
                let d = a | u | v;
                let f = t.index() * 2 + face;
                for q in [a, b, c, d] {
                    out.face_qefs[f] += leafs[q.index()].face_qefs[f];
                }
                // Edges oriented along the v axis on this face
                let edge_index_v = v.index() * 4 + face * 2 + 1;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[edge_index_v].into();
                out.face_qefs[f] +=
                    leafs[b.index()].intersections[edge_index_v].into();

                // Edges oriented along the u axis on this face
                let edge_index_u = v.index() * 4 + face * 2 + 1;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[edge_index_u].into();
                out.face_qefs[f] +=
                    leafs[c.index()].intersections[edge_index_u].into();
            }
        }

        // Accumulate center QEFs
        for t in [X, Y, Z] {
            let u = t.next();
            let v = t.next();

            // Accumulate the four inner face QEFs
            let a = Corner::new(0);
            let b = a | u;
            let c = a | v;
            let d = a | u | v;
            for q in [a, b, c, d] {
                out.center_qef += leafs[q.index()].face_qefs[t.index() * 2 + 1];
            }

            // Edges oriented along the u axis
            out.center_qef +=
                leafs[a.index()].intersections[u.index() * 4 + 3].into();
            out.center_qef +=
                leafs[b.index()].intersections[u.index() * 4 + 3].into();

            // We skip edges oriented on the v axis, because they'll be counted
            // by one of the other iterations through the loop
        }
        for leaf in leafs {
            out.center_qef += leaf.center_qef;
        }

        // Accumulate minimum QEF error among valid child QEFs
        out.qef_err = f32::INFINITY;
        for e in leafs.iter().map(|q| q.qef_err).filter(|&e| e >= 0.0) {
            out.qef_err = out.qef_err.min(e);
        }

        // Build mask
        for (i, leaf) in leafs.iter().enumerate() {
            out.mask |= leaf.mask & (1 << i);
        }

        out
    }

    /// Solves the combined QEF
    pub fn solve(&self) -> (CellVertex, f32) {
        let mut qef = self.center_qef;
        for &i in &self.intersections {
            qef += i.into();
        }
        for &f in &self.face_qefs {
            qef += f;
        }
        qef.solve()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        context::bound::{self, BoundContext, BoundNode},
        eval::{EzShape, MathShape},
        mesh::types::{Edge, X, Y, Z},
        shape::Bounds,
        vm::VmShape,
    };
    use nalgebra::Vector3;
    use std::collections::BTreeMap;

    const DEPTH0_SINGLE_THREAD: Settings = Settings {
        min_depth: 0,
        max_depth: 0,
        threads: 0,
        bounds: Bounds {
            center: Vector3::new(0.0, 0.0, 0.0),
            size: 1.0,
        },
    };
    const DEPTH1_SINGLE_THREAD: Settings = Settings {
        min_depth: 1,
        max_depth: 1,
        threads: 0,
        bounds: Bounds {
            center: Vector3::new(0.0, 0.0, 0.0),
            size: 1.0,
        },
    };

    fn sphere(
        ctx: &BoundContext,
        center: [f32; 3],
        radius: f32,
    ) -> bound::BoundNode {
        let (x, y, z) = ctx.axes();
        ((x - center[0]).square()
            + (y - center[1]).square()
            + (z - center[2]).square())
        .sqrt()
            - radius
    }

    fn cube(
        ctx: &BoundContext,
        bx: [f32; 2],
        by: [f32; 2],
        bz: [f32; 2],
    ) -> BoundNode {
        let (x, y, z) = ctx.axes();
        let x_bounds = (bx[0] - x.clone()).max(x - bx[1]);
        let y_bounds = (by[0] - y.clone()).max(y - by[1]);
        let z_bounds = (bz[0] - z.clone()).max(z - bz[1]);
        x_bounds.max(y_bounds).max(z_bounds)
    }

    #[test]
    fn test_cube_edge() {
        const EPSILON: f32 = 1e-3;
        let ctx = BoundContext::new();
        let f = 2.0;
        let cube = cube(&ctx, [-f, f], [-f, 0.3], [-f, 0.6]);
        // This should be a cube with a single edge running through the root
        // node of the octree, with an edge vertex at [0, 0.3, 0.6]
        let shape: VmShape = cube.convert();
        let octree = Octree::build(&shape, DEPTH0_SINGLE_THREAD);
        assert_eq!(octree.verts.len(), 5);
        let v = octree.verts[0].pos;
        let expected = nalgebra::Vector3::new(0.0, 0.3, 0.6);
        assert!(
            (v - expected).norm() < EPSILON,
            "bad edge vertex {v:?}; expected {expected:?}"
        );
    }

    fn cone(
        ctx: &BoundContext,
        corner: nalgebra::Vector3<f32>,
        tip: nalgebra::Vector3<f32>,
        radius: f32,
    ) -> BoundNode {
        let dir = tip - corner;
        let length = dir.norm();
        let dir = dir.normalize();

        let corner = corner.map(|v| ctx.constant(v as f64));
        let dir = dir.map(|v| ctx.constant(v as f64));

        let (x, y, z) = ctx.axes();
        let point = nalgebra::Vector3::new(x, y, z);
        let offset = point.clone() - corner.clone();

        // a is the distance along the corner-tip direction
        let a = offset.x.clone() * dir.x.clone()
            + offset.y.clone() * dir.y.clone()
            + offset.z.clone() * dir.z.clone();

        // Position of the nearest point on the corner-tip axis
        let a_pos = corner + dir * a.clone();

        // b is the orthogonal distance
        let offset = point - a_pos;
        let b = (offset.x.clone().square()
            + offset.y.clone().square()
            + offset.z.clone().square())
        .sqrt();

        b - radius * (1.0 - a / length)
    }

    #[test]
    fn test_mesh_basic() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.2);
        let shape: VmShape = shape.convert();

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&shape, DEPTH0_SINGLE_THREAD);
        assert_eq!(octree.cells.len(), 8); // we always build at least 8 cells
        assert_eq!(Cell::Empty, octree.cells[0].into(),);
        assert_eq!(octree.verts.len(), 0);

        let empty_mesh = octree.walk_dual(DEPTH0_SINGLE_THREAD);
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&shape, DEPTH1_SINGLE_THREAD);
        assert_eq!(octree.cells.len(), 16); // we always build at least 8 cells
        assert_eq!(
            Cell::Branch {
                index: 8,
                thread: 0
            },
            octree.cells[0].into()
        );

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8, "incorrect vertex count");

        // Each cell is a leaf with 4 vertices (3 edges, 1 center)
        for o in &octree.cells[8..] {
            let Cell::Leaf(Leaf { index, mask }) = (*o).into() else {
                panic!()
            };
            assert_eq!(mask.count_ones(), 1);
            assert_eq!(index % 4, 0);
        }

        let sphere_mesh = octree.walk_dual(DEPTH1_SINGLE_THREAD);
        assert!(sphere_mesh.vertices.len() > 1);
        assert!(!sphere_mesh.triangles.is_empty());
    }

    #[test]
    fn test_sphere_verts() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.2);

        let shape: VmShape = shape.convert();
        let octree = Octree::build(&shape, DEPTH1_SINGLE_THREAD);
        let sphere_mesh = octree.walk_dual(DEPTH1_SINGLE_THREAD);

        let mut edge_count = 0;
        for v in &sphere_mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);
            if edge_sum == 1 {
                assert!(
                    (v.norm() - 0.2).abs() < 2.0 / u16::MAX as f32,
                    "edge vertex {v:?} is not at radius 0.2"
                );
                edge_count += 1;
            } else {
                // The sphere looks like a box at this sampling depth, since the
                // edge intersections are all axis-aligned; we'll check that
                // corner vertices are all at [±0.2, ±0.2, ±0.2]
                assert!(
                    (v.abs() - nalgebra::Vector3::new(0.2, 0.2, 0.2)).norm()
                        < 2.0 / 65535.0,
                    "cell vertex {v:?} is not at expected position"
                );
            }
        }
        assert_eq!(edge_count, 6);
    }

    #[test]
    fn test_sphere_manifold() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.85);
        let shape: VmShape = shape.convert();

        for threads in [0, 8] {
            let settings = Settings {
                min_depth: 5,
                max_depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, settings);
            let sphere_mesh = octree.walk_dual(settings);
            sphere_mesh
                .write_stl(
                    &mut std::fs::File::create(format!("sphere{threads}.stl"))
                        .unwrap(),
                )
                .unwrap();

            if let Err(e) = check_for_vertex_dupes(&sphere_mesh) {
                panic!("{e} (with {threads} threads)");
            }
            if let Err(e) = check_for_edge_matching(&sphere_mesh) {
                panic!("{e} (with {threads} threads)");
            }
        }
    }

    #[test]
    fn test_cube_verts() {
        let ctx = BoundContext::new();
        let shape = cube(&ctx, [-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]);

        let shape: VmShape = shape.convert();
        let octree = Octree::build(&shape, DEPTH1_SINGLE_THREAD);
        let mesh = octree.walk_dual(DEPTH1_SINGLE_THREAD);
        const EPSILON: f32 = 2.0 / u16::MAX as f32;
        assert!(!mesh.vertices.is_empty());
        for v in &mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);

            if edge_sum == 1 {
                assert!(
                    (x_edge
                        && ((v.x - -0.1).abs() < EPSILON
                            || (v.x - 0.6).abs() < EPSILON))
                        || (y_edge
                            && ((v.y - -0.2).abs() < EPSILON
                                || (v.y - 0.75).abs() < EPSILON))
                        || (z_edge
                            && ((v.z - -0.3).abs() < EPSILON
                                || (v.z - 0.4).abs() < EPSILON)),
                    "bad edge position {v:?}"
                );
            } else {
                assert!(
                    ((v.x - -0.1).abs() < EPSILON
                        || (v.x - 0.6).abs() < EPSILON)
                        && ((v.y - -0.2).abs() < EPSILON
                            || (v.y - 0.75).abs() < EPSILON)
                        && ((v.z - -0.3).abs() < EPSILON
                            || (v.z - 0.4).abs() < EPSILON),
                    "bad vertex position {v:?}"
                );
            }
        }
    }

    #[test]
    fn test_plane_center() {
        const EPSILON: f32 = 1e-3;
        let ctx = BoundContext::new();
        for dx in [0.0, 0.25, -0.25, 2.0, -2.0] {
            for dy in [0.0, 0.25, -0.25, 2.0, -2.0] {
                for offset in [0.0, -0.2, 0.2] {
                    let (x, y, z) = ctx.axes();
                    let f = x * dx + y * dy + z + offset;
                    let shape: VmShape = f.convert();
                    let octree = Octree::build(&shape, DEPTH0_SINGLE_THREAD);

                    assert_eq!(octree.cells.len(), 8);
                    let pos = octree.verts[0].pos;
                    let mut mass_point = nalgebra::Vector3::zeros();
                    for v in &octree.verts[1..] {
                        mass_point += v.pos;
                    }
                    mass_point /= (octree.verts.len() - 1) as f32;
                    assert!(
                        (pos - mass_point).norm() < EPSILON,
                        "bad vertex position at dx: {dx}, dy: {dy}, \
                         offset: {offset} => {pos:?} != {mass_point:?}"
                    );
                    let mut eval = VmShape::new_point_eval();
                    let tape = shape.ez_point_tape();
                    for v in &octree.verts {
                        let v = v.pos;
                        let (r, _) =
                            eval.eval(&tape, v.x, v.y, v.z, &[]).unwrap();
                        assert!(r.abs() < EPSILON, "bad value at {v:?}: {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cone_vert() {
        // Test both in-cell and out-of-cell cone vertices
        for tip in [
            nalgebra::Vector3::new(0.2, 0.3, 0.4),
            nalgebra::Vector3::new(1.2, 1.3, 1.4),
        ] {
            let ctx = BoundContext::new();
            let corner = nalgebra::Vector3::new(-1.0, -1.0, -1.0);
            let shape = cone(&ctx, corner, tip, 0.1);
            let shape: VmShape = shape.convert();

            let mut eval = VmShape::new_point_eval();
            let tape = shape.ez_point_tape();
            let (v, _) = eval.eval(&tape, tip.x, tip.y, tip.z, &[]).unwrap();
            assert!(v.abs() < 1e-6, "bad tip value: {v}");
            let (v, _) =
                eval.eval(&tape, corner.x, corner.y, corner.z, &[]).unwrap();
            assert!(v < 0.0, "bad corner value: {v}");

            let octree = Octree::build(&shape, DEPTH0_SINGLE_THREAD);
            assert_eq!(octree.cells.len(), 8);
            assert_eq!(octree.verts.len(), 4);

            let pos = octree.verts[0].pos;
            assert!(
                (pos - tip).norm() < 1e-3,
                "bad vertex position: expected {tip:?}, got {pos:?}"
            );
        }
    }

    #[test]
    fn test_mesh_manifold() {
        for threads in [0, 8] {
            for i in 0..256 {
                let ctx = BoundContext::new();
                let mut shape = vec![];
                for j in Corner::iter() {
                    if i & (1 << j.index()) != 0 {
                        shape.push(sphere(
                            &ctx,
                            [
                                if j & X { 0.5 } else { 0.0 },
                                if j & Y { 0.5 } else { 0.0 },
                                if j & Z { 0.5 } else { 0.0 },
                            ],
                            0.1,
                        ));
                    }
                }
                let Some(start) = shape.pop() else { continue };
                let shape = shape.into_iter().fold(start, |acc, s| acc.min(s));

                // Now, we have our shape, which is 0-8 spheres placed at the
                // corners of the cell spanning [0, 0.25]
                let shape: VmShape = shape.convert();
                let settings = Settings {
                    min_depth: 2,
                    max_depth: 2,
                    threads,
                    ..Default::default()
                };
                let octree = Octree::build(&shape, settings);

                let mesh = octree.walk_dual(settings);
                if i != 0 && i != 255 {
                    assert!(!mesh.vertices.is_empty());
                    assert!(!mesh.triangles.is_empty());
                }

                if let Err(e) = check_for_vertex_dupes(&mesh) {
                    panic!("mask {i:08b} has {e}");
                }
                if let Err(e) = check_for_edge_matching(&mesh) {
                    panic!("mask {i:08b} has {e}");
                }
            }
        }
    }

    #[test]
    fn test_collapsible() {
        let ctx = BoundContext::new();

        fn builder(
            shape: BoundNode,
            settings: Settings,
        ) -> OctreeBuilder<VmShape> {
            let shape: VmShape = shape.convert();
            let eval = Arc::new(EvalGroup::new(shape));
            let mut out = OctreeBuilder::new();
            out.recurse(&eval, CellIndex::default(), settings);
            out
        }

        let shape = sphere(&ctx, [0.0; 3], 0.1);
        let octree = builder(shape, DEPTH1_SINGLE_THREAD);
        assert!(!octree.collapsible(8));

        let shape = sphere(&ctx, [-1.0; 3], 0.1);
        let octree = builder(shape, DEPTH1_SINGLE_THREAD);
        assert!(octree.collapsible(8));

        let shape = sphere(&ctx, [-1.0, 0.0, 1.0], 0.1);
        let octree = builder(shape, DEPTH1_SINGLE_THREAD);
        assert!(!octree.collapsible(8));

        let a = sphere(&ctx, [-1.0; 3], 0.1);
        let b = sphere(&ctx, [1.0; 3], 0.1);
        let shape = a.min(b);
        let octree = builder(shape, DEPTH1_SINGLE_THREAD);
        assert!(!octree.collapsible(8));
    }

    #[test]
    fn test_empty_collapse() {
        // Make a very smol sphere that won't be sampled
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.1; 3], 0.05);
        let tape: VmShape = shape.convert();
        for threads in [0, 4] {
            let settings = Settings {
                min_depth: 1,
                max_depth: 1,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, settings);
            assert_eq!(
                octree.cells[0],
                Cell::Empty.into(),
                "failed to collapse octree with {threads} threads"
            );
        }
    }

    #[test]
    fn test_colonnade_manifold() {
        const COLONNADE: &str = include_str!("../../../models/colonnade.vm");
        let (ctx, root) =
            crate::Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();
        for threads in [0, 8] {
            let settings = Settings {
                min_depth: 5,
                max_depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, settings);
            let mesh = octree.walk_dual(settings);
            // Note: the model has duplicate vertices!
            if let Err(e) = check_for_edge_matching(&mesh) {
                panic!("colonnade model has {e}");
            }
        }
    }

    fn check_for_vertex_dupes(mesh: &Mesh) -> Result<(), String> {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            if verts[i - 1] == verts[i] {
                return Err(format!("duplicate vertices at {}", verts[i]));
            }
        }
        Ok(())
    }

    fn check_for_edge_matching(mesh: &Mesh) -> Result<(), String> {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                if t.x == t.y || t.y == t.z || t.x == t.z {
                    return Err("triangle with duplicate edges".to_string());
                }
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            if i != 1 {
                return Err(format!(
                    "duplicate edge ({a}, {b}) between {:?} {:?}",
                    mesh.vertices[a], mesh.vertices[b]
                ));
            }
            if !edges.contains_key(&(b, a)) {
                return Err("unpaired edges".to_owned());
            }
        }
        Ok(())
    }

    #[test]
    fn test_qef_merging() {
        let mut hermite = LeafHermiteData::new();

        // Add a dummy intersection with a non-zero normal; we'll be counting
        // mass points to check that the merging went smoothly
        let grad = nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0);
        let pos = nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
        hermite.intersections.fill(LeafIntersection { pos, grad });

        // Ensure that only one of the two sub-edges per edge contains an
        // intersection, because that's checked for in `LeafHermiteData::merge`
        let mut hermites = [hermite; 8];
        for i in 0..12 {
            let e = Edge::new(i);
            let (_start, end) = e.corners();
            hermites[end.index()].intersections[i as usize] =
                LeafIntersection::default();
        }

        let merged = LeafHermiteData::merge(hermites);
        for i in merged.intersections {
            assert_eq!(i.grad, grad);
            assert_eq!(i.pos, pos);
        }
        // Each face in the merged cell should include the accumulation of four
        // edges from lower cells (but nothing more, because lower cells didn't
        // have face QEFs populated)
        for (i, f) in merged.face_qefs.iter().enumerate() {
            assert_eq!(
                f.mass_point().w,
                4.0,
                "bad accumulated QEF on face {i}"
            );
        }
        assert_eq!(
            merged.center_qef.mass_point().w,
            6.0,
            "bad accumulated QEF in center"
        );
    }

    #[test]
    fn test_qef_near_planar() {
        let ctx = BoundContext::new();
        let shape = sphere(&ctx, [0.0; 3], 0.75);

        let shape: VmShape = shape.convert();
        let settings = Settings {
            min_depth: 4,
            max_depth: 4,
            threads: 0,
            ..Default::default()
        };

        let octree = Octree::build(&shape, settings).walk_dual(settings);
        for v in octree.vertices.iter() {
            let n = v.norm();
            assert!(n > 0.7 && n < 0.8, "invalid vertex at {v:?}: {n}");
        }
    }
}
