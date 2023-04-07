//! An octree data structure and implementation of Manifold Dual Contouring

use super::{
    builder::MeshBuilder,
    cell::{Cell, CellData, CellIndex, CellVertex, Leaf},
    dc::{DcBuilder, DcWorker},
    gen::CELL_TO_VERT_TO_EDGES,
    types::{Axis, Corner},
    worker::Worker,
    Mesh, Settings,
};
use crate::eval::{
    float_slice::{FloatSliceEvalData, FloatSliceEvalStorage},
    grad_slice::{GradSliceEvalData, GradSliceEvalStorage},
    interval::{IntervalEvalData, IntervalEvalStorage},
    tape,
    types::Interval,
    Family, FloatSliceEval, GradSliceEval, IntervalEval, Tape,
};
use once_cell::sync::OnceCell;
use std::sync::Arc;

/// Helper struct to contain a set of matched evaluators
///
/// Note that this is `Send + Sync` and can be used with shared references!
pub struct EvalGroup<I: Family> {
    pub tape: Tape<I>,

    // TODO: passing around an `Arc<EvalGroup>` ends up with two layers of
    // indirection (since the evaluators also contain `Arc`); could we flatten
    // them out?
    pub interval: OnceCell<IntervalEval<I>>,
    pub float_slice: OnceCell<FloatSliceEval<I>>,
    pub grad_slice: OnceCell<GradSliceEval<I>>,
}

impl<I: Family> EvalGroup<I> {
    fn new(tape: Tape<I>) -> Self {
        Self {
            tape,
            interval: OnceCell::new(),
            float_slice: OnceCell::new(),
            grad_slice: OnceCell::new(),
        }
    }
    fn interval(
        &self,
        s: &mut Vec<IntervalEvalStorage<I>>,
    ) -> &IntervalEval<I> {
        self.interval.get_or_init(|| {
            self.tape.new_interval_evaluator_with_storage(
                s.pop().unwrap_or_default(),
            )
        })
    }
    fn float_slice(
        &self,
        s: &mut Vec<FloatSliceEvalStorage<I>>,
    ) -> &FloatSliceEval<I> {
        self.float_slice.get_or_init(|| {
            self.tape.new_float_slice_evaluator_with_storage(
                s.pop().unwrap_or_default(),
            )
        })
    }
    fn grad_slice(
        &self,

        s: &mut Vec<GradSliceEvalStorage<I>>,
    ) -> &GradSliceEval<I> {
        self.grad_slice.get_or_init(|| {
            self.tape.new_grad_slice_evaluator_with_storage(
                s.pop().unwrap_or_default(),
            )
        })
    }
}

pub struct EvalData<I: Family> {
    float_data: FloatSliceEvalData<I>,
    grad_data: GradSliceEvalData<I>,
    interval_data: IntervalEvalData<I>,
}

impl<I: Family> Default for EvalData<I> {
    fn default() -> Self {
        Self {
            float_data: Default::default(),
            grad_data: Default::default(),
            interval_data: Default::default(),
        }
    }
}

pub struct EvalStorage<I: Family> {
    pub workspace: tape::Workspace,
    pub tape_storage: Vec<tape::Data>,
    pub float_storage: Vec<FloatSliceEvalStorage<I>>,
    pub interval_storage: Vec<IntervalEvalStorage<I>>,
    pub grad_storage: Vec<GradSliceEvalStorage<I>>,
}

impl<I: Family> EvalStorage<I> {
    pub fn claim(&mut self, mut e: EvalGroup<I>) {
        self.interval_storage
            .extend(e.interval.take().and_then(|s| s.take()));
        self.float_storage
            .extend(e.float_slice.take().and_then(|s| s.take()));
        self.grad_storage
            .extend(e.grad_slice.take().and_then(|s| s.take()));
        self.tape_storage.extend(e.tape.take());
    }
}

impl<I: Family> Default for EvalStorage<I> {
    fn default() -> Self {
        Self {
            workspace: Default::default(),
            tape_storage: Default::default(),
            float_storage: Default::default(),
            grad_storage: Default::default(),
            interval_storage: Default::default(),
        }
    }
}

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    /// The top two bits determine cell types
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

impl Default for Octree {
    fn default() -> Self {
        Self::new()
    }
}

impl Octree {
    /// Builds a new octree, which allocates data for 8 root cells
    pub(crate) fn new() -> Self {
        Self {
            cells: vec![Cell::Invalid.into(); 8],
            verts: vec![],
        }
    }

    /// Builds a new empty octree
    pub(crate) fn empty() -> Self {
        Self {
            cells: vec![],
            verts: vec![],
        }
    }

    /// Records the given cell into the provided index
    ///
    /// The index must be valid already; this does not modify the cells vector.
    ///
    /// # Panics
    /// If the index exceeds the bounds of the cell vector, or the cell is
    /// already populated.
    pub(crate) fn record(&mut self, index: usize, cell: CellData) {
        debug_assert_eq!(self.cells[index], Cell::Invalid.into());
        self.cells[index] = cell;
    }

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
    pub fn build<I: Family>(tape: &Tape<I>, settings: Settings) -> Self {
        let eval = Arc::new(EvalGroup::new(tape.clone()));

        if settings.threads == 0 {
            let mut out = Self {
                cells: vec![Cell::Invalid.into(); 8],
                verts: vec![],
            };
            out.recurse(
                &eval,
                &mut EvalData::default(),
                &mut EvalStorage::default(),
                CellIndex::default(),
                settings,
            );
            out
        } else {
            Worker::scheduler(eval, settings)
        }
    }

    /// Evaluates a single cell in the octree
    ///
    /// Leaf data is stored in `self.verts`; cell results are **not** written
    /// back to the `cells` array, because the cell may be rooted in a different
    /// octree (e.g. on another thread).
    pub(crate) fn eval_cell<I: Family>(
        &mut self,
        eval: &Arc<EvalGroup<I>>,
        data: &mut EvalData<I>,
        storage: &mut EvalStorage<I>,
        cell: CellIndex,
        settings: Settings,
    ) -> CellResult<I> {
        let (i, r) = eval
            .interval(&mut storage.interval_storage)
            .eval_with(cell.x, cell.y, cell.z, &[], &mut data.interval_data)
            .unwrap();
        if i.upper() < 0.0 {
            CellResult::Full
        } else if i.lower() > 0.0 {
            CellResult::Empty
        } else {
            let sub_tape = if I::simplify_tree_during_meshing(cell.depth) {
                r.map(|r| {
                    Arc::new(EvalGroup::new(
                        r.simplify_with(
                            &mut storage.workspace,
                            storage.tape_storage.pop().unwrap_or_default(),
                        )
                        .unwrap(),
                    ))
                })
            } else {
                None
            };
            if cell.depth == settings.min_depth as usize {
                let eval = sub_tape.unwrap_or_else(|| eval.clone());
                CellResult::Leaf(self.leaf(&eval, data, storage, cell))
            } else {
                CellResult::Recurse(sub_tape.unwrap_or_else(|| eval.clone()))
            }
        }
    }

    /// Recurse down the octree, building the given cell
    fn recurse<I: Family>(
        &mut self,
        eval: &Arc<EvalGroup<I>>,
        data: &mut EvalData<I>,
        storage: &mut EvalStorage<I>,
        cell: CellIndex,
        settings: Settings,
    ) {
        match self.eval_cell(eval, data, storage, cell, settings) {
            CellResult::Empty => self.cells[cell.index] = Cell::Empty.into(),
            CellResult::Full => self.cells[cell.index] = Cell::Full.into(),
            CellResult::Leaf(leaf) => {
                self.cells[cell.index] = Cell::Leaf(leaf).into()
            }
            CellResult::Recurse(eval) => {
                let index = self.cells.len();
                for _ in Corner::iter() {
                    self.cells.push(Cell::Invalid.into());
                }
                self.cells[cell.index] =
                    Cell::Branch { index, thread: 0 }.into();
                for i in Corner::iter() {
                    let cell = cell.child(index, i);
                    self.recurse(&eval, data, storage, cell, settings);
                }
                // Try to recycle tape storage
                if let Ok(e) = Arc::try_unwrap(eval) {
                    storage.claim(e);
                }
            }
        }
    }

    /// Evaluates the given leaf
    fn leaf<I: Family>(
        &mut self,
        eval: &EvalGroup<I>,
        data: &mut EvalData<I>,
        storage: &mut EvalStorage<I>,
        cell: CellIndex,
    ) -> Leaf {
        let float_eval = eval.float_slice(&mut storage.float_storage);

        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::iter() {
            let (x, y, z) = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        let out = float_eval
            .eval_with(&xs, &ys, &zs, &[], &mut data.float_data)
            .unwrap();
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, &v)| v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

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

                    let pos = cell.pos(pos.map(|v| v as i32));
                    xs[i] = pos.x;
                    ys[i] = pos.y;
                    zs[i] = pos.z;
                    i += 1;
                }
            }
            debug_assert_eq!(i, EDGE_SEARCH_SIZE * edge_count);

            // Do the actual evaluation
            let out = float_eval
                .eval_with(xs, ys, zs, &[], &mut data.float_data)
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
            let pos = cell.pos(xyz.map(|i| i as i32));
            xs[i] = pos.x;
            ys[i] = pos.y;
            zs[i] = pos.z;
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grad_eval = eval.grad_slice(&mut storage.grad_storage);
        let grads = grad_eval
            .eval_with(xs, ys, zs, &[], &mut data.grad_data)
            .unwrap();

        // Now, we're going to solve a quadratic error function for every vertex
        // to position it at the right place.  This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF
        // math and references to primary sources.
        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        for vs in CELL_TO_VERT_TO_EDGES[mask as usize].iter() {
            let mut ata = nalgebra::Matrix3::zeros();
            let mut atb = nalgebra::Vector3::zeros();
            let mut btb = 0.0;

            let mut mass_point = nalgebra::Vector3::zeros();
            for _ in 0..vs.len() {
                let pos = nalgebra::Vector3::new(xs[i], ys[i], zs[i]);
                mass_point += pos;

                let grad = grads[i];
                let norm = nalgebra::Vector3::new(grad.dx, grad.dy, grad.dz)
                    .normalize();

                // TODO: correct for non-zero distance value?

                ata += norm * norm.transpose();
                atb += norm * norm.dot(&pos);
                btb += norm.dot(&pos).powi(2);
                i += 1;
            }
            // Minimize towards mass point of intersections, which requires
            // substituting B' = B - Ac in the QEF equations.
            let center = mass_point / vs.len() as f32;
            btb += ((center.transpose() * ata - atb.transpose()) * center
                - center.transpose() * atb)[0];
            atb -= ata * center;

            let svd = nalgebra::linalg::SVD::new(ata, true, true);
            // "Dual Contouring: The Secret Sauce" recomments a threshold of 0.1
            // when using normalized gradients, but I've found that fails on
            // things like the cone model.  Instead, we'll be a little more
            // clever: we'll pick the smallest epsilon that keeps the feature in
            // the cell without dramatically increasing QEF error.
            //
            // TODO: iterating by epsilons is a _little_ silly, because what we
            // actually care about is turning off the 0/1/2/3 lowest eigenvalues
            // in the solution matrix.
            const EPSILONS: &[f32] = &[1e-4, 1e-3, 1e-2];
            let mut prev = None;
            for (i, &epsilon) in EPSILONS.iter().enumerate() {
                let sol = svd.solve(&atb, epsilon);
                let pos = sol.map(|c| c + center).unwrap_or(center);
                let err = (pos.transpose() * ata * pos
                    - 2.0 * pos.transpose() * atb)[0]
                    + btb;

                // If this epsilon dramatically increases the error, then we'll
                // assume that the previous (out-of-cell) vertex was genuine and
                // use it.
                if let Some((_, prev_pos)) =
                    prev.filter(|(prev_err, _)| err > prev_err * 2.0)
                {
                    verts.push(prev_pos);
                    break;
                }

                // If the matrix solution is in the cell, then we assume the
                // solution is good; we _also_ stop iterating if this is the
                // last possible chance.
                let pos = cell.relative(pos);
                if i == EPSILONS.len() - 1 || pos.valid() {
                    verts.push(pos);
                    break;
                }
                prev = Some((err, pos));
            }
        }

        let index = self.verts.len();
        self.verts.extend(verts.into_iter());
        self.verts
            .extend(intersections.into_iter().map(|pos| CellVertex {
                pos: pos.map(|i| i as i32),
            }));
        Leaf { mask, index }
    }

    /// Recursively walks the dual of the octree, building a mesh
    pub fn walk_dual(&self, settings: Settings) -> Mesh {
        let x = Interval::new(-1.0, 1.0);
        let y = Interval::new(-1.0, 1.0);
        let z = Interval::new(-1.0, 1.0);
        let mut mesh = MeshBuilder::default();

        if settings.threads == 0 {
            mesh.cell(
                &self,
                CellIndex {
                    index: 0,
                    x,
                    y,
                    z,
                    depth: 0,
                },
            );
            mesh.take()
        } else {
            DcWorker::scheduler(self, settings.threads)
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

        match self.cells[cell.index].into() {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index, .. } => cell.child(index, child),
            Cell::Invalid => panic!(),
        }
    }

    pub(crate) fn is_leaf(&self, cell: CellIndex) -> bool {
        match self.cells[cell.index].into() {
            Cell::Leaf(..) | Cell::Full | Cell::Empty => true,
            Cell::Branch { .. } => false,
            Cell::Invalid => panic!(),
        }
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
        let cells = &self.cells[root..root + 8];
        let mut mask = 0;
        for (i, &c) in cells.iter().enumerate() {
            let b = match c.into() {
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
        assert_ne!(mask, 255);
        assert_ne!(mask, 0);
        // TODO: this check may not be necessary, because we're doing *manifold*
        // dual contouring; the collapsed cell can have multiple vertices.
        CELL_TO_VERT_TO_EDGES[mask as usize].len() == 1
    }
}

/// Result of a single cell evaluation
pub enum CellResult<I: Family> {
    Empty,
    Full,
    Leaf(Leaf),
    Recurse(Arc<EvalGroup<I>>),
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        context::bound::{self, BoundContext, BoundNode},
        mesh::types::{X, Y, Z},
    };
    use std::collections::BTreeMap;

    const DEPTH0_SINGLE_THREAD: Settings = Settings {
        min_depth: 0,
        max_depth: 0,
        threads: 0,
    };
    const DEPTH1_SINGLE_THREAD: Settings = Settings {
        min_depth: 1,
        max_depth: 1,
        threads: 0,
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
        let tape = cube.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, DEPTH0_SINGLE_THREAD);
        assert_eq!(octree.verts.len(), 5);
        let v = CellIndex::default().pos(octree.verts[0]);
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
        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&tape, DEPTH0_SINGLE_THREAD);
        assert_eq!(octree.cells.len(), 8); // we always build at least 8 cells
        assert_eq!(
            Cell::Leaf(Leaf { mask: 0, index: 0 },),
            octree.cells[0].into(),
        );
        assert_eq!(octree.verts.len(), 0);
        // TODO: should we transform this into an Empty?

        let empty_mesh = octree.walk_dual(DEPTH0_SINGLE_THREAD);
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&tape, DEPTH1_SINGLE_THREAD);
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
            let Cell::Leaf ( Leaf { index, mask}) = (*o).into()
                else { panic!() };
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

        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, DEPTH1_SINGLE_THREAD);
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
        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();

        for threads in [0, 8] {
            let settings = Settings {
                min_depth: 5,
                max_depth: 5,
                threads,
            };
            let octree = Octree::build(&tape, settings);
            let sphere_mesh = octree.walk_dual(settings);
            /*
            sphere_mesh
                .write_stl(
                    &mut std::fs::File::create(format!("out{threads}.stl"))
                        .unwrap(),
                )
                .unwrap();
            */

            if let Err(e) = check_for_vertex_dupes(&sphere_mesh) {
                panic!("{e}");
            }
            if let Err(e) = check_for_edge_matching(&sphere_mesh) {
                panic!("{e}");
            }
        }
    }

    #[test]
    fn test_cube_verts() {
        let ctx = BoundContext::new();
        let shape = cube(&ctx, [-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]);

        let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
        let octree = Octree::build(&tape, DEPTH1_SINGLE_THREAD);
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
                    let tape = f.get_tape::<crate::vm::Eval>().unwrap();
                    let octree = Octree::build(&tape, DEPTH0_SINGLE_THREAD);

                    assert_eq!(octree.cells.len(), 8);
                    let pos = CellIndex::default().pos(octree.verts[0]);
                    let mut mass_point = nalgebra::Vector3::zeros();
                    for v in &octree.verts[1..] {
                        mass_point += CellIndex::default().pos(*v);
                    }
                    mass_point /= (octree.verts.len() - 1) as f32;
                    assert!(
                        (pos - mass_point).norm() < EPSILON,
                        "bad vertex position at dx: {dx}, dy: {dy}, \
                         offset: {offset} => {pos:?} != {mass_point:?}"
                    );
                    let eval = tape.new_point_evaluator();
                    for v in &octree.verts {
                        let v = CellIndex::default().pos(*v);
                        let (r, _) = eval.eval(v.x, v.y, v.z, &[]).unwrap();
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
            let tape = shape.get_tape::<crate::vm::Eval>().unwrap();

            let eval = tape.new_point_evaluator();
            let (v, _) = eval.eval(tip.x, tip.y, tip.z, &[]).unwrap();
            assert!(v.abs() < 1e-6, "bad tip value: {v}");
            let (v, _) = eval.eval(corner.x, corner.y, corner.z, &[]).unwrap();
            assert!(v < 0.0, "bad corner value: {v}");

            let octree = Octree::build(&tape, DEPTH0_SINGLE_THREAD);
            assert_eq!(octree.cells.len(), 8);
            assert_eq!(octree.verts.len(), 4);

            let pos = CellIndex::default().pos(octree.verts[0]);
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
                let tape = shape.get_tape::<crate::vm::Eval>().unwrap();
                let settings = Settings {
                    min_depth: 2,
                    max_depth: 2,
                    threads,
                };
                let octree = Octree::build(&tape, settings);

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

    fn check_for_vertex_dupes(mesh: &Mesh) -> Result<(), String> {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            if verts[i - 1] == verts[i] {
                return Err("duplicate vertices".to_owned());
            }
        }
        Ok(())
    }

    fn check_for_edge_matching(mesh: &Mesh) -> Result<(), String> {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            if i != 1 {
                return Err(format!("duplicate edge ({a}, {b})"));
            }
            if !edges.contains_key(&(b, a)) {
                return Err("unpaired edges".to_owned());
            }
        }
        Ok(())
    }
}
