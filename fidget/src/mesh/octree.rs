//! An octree data structure and implementation of Manifold Dual Contouring

use super::{
    builder::MeshBuilder,
    cell::{Cell, CellIndex, CellVertex, Leaf},
    frame::Frame,
    gen::CELL_TO_VERT_TO_EDGES,
    qef::QuadraticErrorSolver,
    types::{Axis, CellMask, Corner, Edge},
    Mesh, Settings,
};
use crate::{
    eval::Function,
    render::{RenderHandle, RenderHints, ThreadPool},
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::Grad,
};
use std::collections::VecDeque;

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    pub(crate) root: Cell<3>,
    pub(crate) cells: Vec<[Cell<3>; 8]>,

    /// Cell vertices, given as positions within the cell
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    pub(crate) verts: Vec<CellVertex<3>>,
}

impl Octree {
    /// Builds an octree with the root marked as invalid
    pub(crate) fn new() -> Self {
        Octree {
            root: Cell::Invalid,
            cells: vec![],
            verts: vec![],
        }
    }

    /// Builds an octree to the given depth, with user-provided variables
    ///
    /// The shape is evaluated on the region specified by `settings.bounds`.
    pub fn build_with_vars<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        vars: &ShapeVars<f32>,
        settings: Settings,
    ) -> Self {
        // Transform the shape given our world-to-model matrix
        let t = settings.view.world_to_model();
        if t == nalgebra::Matrix4::identity() {
            Self::build_inner(shape, vars, settings)
        } else {
            let shape = shape.clone().apply_transform(t);
            let mut out = Self::build_inner(&shape, vars, settings);

            // Apply the transform from [-1, +1] back to model space
            for v in &mut out.verts {
                let p: nalgebra::Point3<f32> = v.pos.into();
                let q = t.transform_point(&p);
                v.pos = q.coords;
            }
            out
        }
    }

    /// Builds an octree to the given depth
    ///
    /// If the shape uses variables other than `x`, `y`, `z`, then
    /// [`build_with_vars`](Octree::build_with_vars) should be used instead (and
    /// this function will return an error).
    ///
    /// The shape is evaluated on the region specified by `settings.bounds`.
    pub fn build<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        settings: Settings,
    ) -> Self {
        Self::build_with_vars(shape, &ShapeVars::new(), settings)
    }

    fn build_inner<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        vars: &ShapeVars<f32>,
        settings: Settings,
    ) -> Self {
        if let Some(threads) = settings.threads {
            Self::build_inner_mt(shape, vars, settings.depth, threads)
        } else {
            let mut eval = RenderHandle::new(shape.clone());
            let mut out = OctreeBuilder::new();
            let mut hermite = LeafHermiteData::default();
            out.recurse(
                &mut eval,
                vars,
                CellIndex::default(),
                settings.depth,
                &mut hermite,
            );
            out.octree
        }
    }

    /// Multithreaded constructor
    fn build_inner_mt<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        vars: &ShapeVars<f32>,
        max_depth: u8,
        threads: &ThreadPool,
    ) -> Self {
        let mut root = Octree::new();
        let mut todo = VecDeque::new();
        todo.push_back(CellIndex::<3>::default());
        let mut fixup = vec![];
        let mut hermites = vec![];

        // We want a number of tasks that's significantly larger than our thread
        // count, so that we can fully saturate all cores even if tasks take
        // different amounts of time.
        let target_count =
            (8usize.pow(u32::from(max_depth))).min(threads.thread_count() * 10);
        while todo.len() < target_count {
            let next = todo.pop_front().unwrap();

            // Reserve new cells for the 8x children
            let index = root.cells.len();
            root.cells.push([Cell::Invalid; 8]);
            hermites.push([LeafHermiteData::default(); 8]);
            for i in Corner::<3>::iter() {
                let cell = next.child(index, i);
                todo.push_back(cell);
            }
            fixup.push((next, index));
        }

        use rayon::prelude::*;

        struct Output {
            cell: CellIndex<3>,
            octree: Octree,
            hermite: LeafHermiteData,
        }
        let mut rh = RenderHandle::new(shape.clone());
        let _ = rh.i_tape(&mut vec![]); // pre-populate interval tape
        let out = threads.run(|| {
            todo.par_iter()
                .map_init(
                    || (OctreeBuilder::new(), rh.clone()),
                    |(builder, eval), cell| {
                        let mut hermite = LeafHermiteData::default();
                        // Patch our cell so that it builds at index 0
                        let local_cell = CellIndex {
                            index: None,
                            ..*cell
                        };
                        builder.recurse(
                            eval,
                            vars,
                            local_cell,
                            max_depth,
                            &mut hermite,
                        );
                        let octree = std::mem::replace(
                            &mut builder.octree,
                            Octree::new(),
                        );
                        Output {
                            octree,
                            cell: *cell,
                            hermite,
                        }
                    },
                )
                .collect::<Vec<_>>()
        });

        // Copy hermite data into arrays, and compute cumulative offsets
        let mut cell_offsets = vec![root.cells.len()];
        let mut vert_offsets = vec![0];
        for o in &out {
            let (i, j) = o.cell.index.unwrap();
            hermites[i][j as usize] = o.hermite;
            let c = cell_offsets.last().unwrap() + o.octree.cells.len();
            cell_offsets.push(c);
            let v = vert_offsets.last().unwrap() + o.octree.verts.len();
            vert_offsets.push(v);
        }
        root.cells.reserve(*cell_offsets.last().unwrap());
        root.verts.reserve(*vert_offsets.last().unwrap());

        for (i, o) in out.into_iter().enumerate() {
            assert_eq!(cell_offsets[i], root.cells.len());
            assert_eq!(vert_offsets[i], root.verts.len());
            let remap_cell = |c| match c {
                Cell::Leaf(Leaf { mask, index }) => Cell::Leaf(Leaf {
                    mask,
                    index: index + vert_offsets[i],
                }),
                Cell::Branch { index } => Cell::Branch {
                    index: index + cell_offsets[i],
                },
                Cell::Full | Cell::Empty => c,
                Cell::Invalid => panic!(),
            };
            root.cells.extend(
                o.octree.cells.into_iter().map(|cs| cs.map(remap_cell)),
            );
            root.verts.extend(o.octree.verts);
            root[o.cell] = remap_cell(o.octree.root);
        }

        // Walk back up the tree, merging cells as we go
        for (cell, index) in fixup.into_iter().rev() {
            let h = hermites[index];
            root[cell] = root.check_done(
                cell,
                index,
                h,
                cell.index
                    .map(|(i, j)| &mut hermites[i][j as usize])
                    .unwrap_or(&mut LeafHermiteData::default()),
            );
        }
        root
    }

    /// Recursively walks the dual of the octree, building a mesh
    pub fn walk_dual(&self, _settings: Settings) -> Mesh {
        let mut mesh = MeshBuilder::default();

        mesh.cell(self, CellIndex::default());
        mesh.take()
    }

    pub(crate) fn is_leaf(&self, cell: CellIndex<3>) -> bool {
        match self[cell] {
            Cell::Leaf(..) | Cell::Full | Cell::Empty => true,
            Cell::Branch { .. } => false,
            Cell::Invalid => panic!(),
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    ///
    /// # Panics
    /// If the cell is [`Invalid`](Cell::Invalid)
    pub(crate) fn child<C: Into<Corner<3>>>(
        &self,
        cell: CellIndex<3>,
        child: C,
    ) -> CellIndex<3> {
        let child = child.into();

        match self[cell] {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => cell.child(index, child),
            Cell::Invalid => panic!(),
        }
    }

    /// Checks the set of 8 children starting at the given index for completion
    ///
    /// If all are empty or full, then pro-actively collapses the cells (freeing
    /// them if they're at the tail end of the array).
    fn check_done(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Cell<3> {
        // Check out the children
        let mut full_count = 0;
        let mut empty_count = 0;
        for i in 0..8 {
            match self.cells[index][i] {
                Cell::Invalid => {
                    panic!("found invalid cell during meshing")
                }
                Cell::Full => {
                    full_count += 1;
                }
                Cell::Empty => {
                    empty_count += 1;
                }
                Cell::Branch { .. } => return Cell::Branch { index },
                Cell::Leaf(Leaf { .. }) => (),
            }
        }

        // If all of the branches are empty or full, then we're going to
        // record an Empty or Full cell in the parent and don't need the
        // 8x children.  Drop them by resizing the array
        let out = if full_count == 8 {
            Cell::Full
        } else if empty_count == 8 {
            Cell::Empty
        } else if let Some(leaf) =
            self.try_collapse(cell, index, hermite_data, hermite)
        {
            Cell::Leaf(leaf)
        } else {
            Cell::Branch { index }
        };

        // If we can collapse the cell, then we'll be recording an Empty / Full
        // / Leaf cell in the parent and don't need the 8x children.
        //
        // We can drop them if they happen to be at the tail end of the
        // octree; otherwise, we'll satisfy ourselves with setting them to
        // invalid.  We will always be at the tail end of the octree during
        // single-threaded evaluation, it's only during merging octrees from
        // multiple threads that we'd have cells midway through the array.
        if !matches!(out, Cell::Branch { .. }) {
            if index == self.cells.len() - 1 {
                self.cells.resize(index, [Cell::Invalid; 8]);
            } else {
                self.cells[index] = [Cell::Invalid; 8];
            }
        }
        out
    }

    /// Try to collapse the given cell
    ///
    /// Writes to `self.verts` and returns the leaf if successful; otherwise
    /// returns `None`.
    fn try_collapse(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Option<Leaf<3>> {
        let mask = self.collapsible(index)?;
        *hermite = LeafHermiteData::merge(hermite_data)?;

        // Empty / full cells should never be produced here.  The
        // only way to get an empty / full cell is for all eight
        // corners to be empty / full; if that was the case, then
        // either:
        //
        // - The interior vertices match, in which case this should
        //   have been collapsed into a single empty / full cell
        // - The interior vertices *do not* match, in which case the
        //   cell should not be marked as collapsible.
        let (pos, new_err) = hermite.solve();
        if new_err >= hermite.qef_err * 2.0 || !cell.bounds.contains(pos) {
            return None;
        }
        // Record the newly-collapsed leaf
        hermite.qef_err = new_err;

        let index = self.verts.len();
        self.verts.push(pos);

        // Install cell intersections, of which there must only
        // be one (since we only collapse manifold cells)
        let edges = &CELL_TO_VERT_TO_EDGES[mask.index()];
        debug_assert_eq!(edges.len(), 1);
        for e in edges[0] {
            let i = hermite.intersections[e.to_undirected().index()];
            self.verts.push(CellVertex { pos: i.pos.xyz() });
        }

        Some(Leaf { mask, index })
    }

    /// Checks whether the set of 8 cells beginning at `root` can be collapsed.
    ///
    /// Only topology is checked, based on the three predicates from "Dual
    /// Contouring of Hermite Data" (Ju et al, 2002), §4.1
    pub(crate) fn collapsible(&self, root: usize) -> Option<CellMask<3>> {
        // Copy cells to a local variable for simplicity
        let cells = self.cells[root];

        let mut mask = 0;
        for (i, &c) in cells.iter().enumerate() {
            let b = match c {
                Cell::Leaf(Leaf { mask, .. }) => {
                    if CELL_TO_VERT_TO_EDGES[mask.index()].len() > 1 {
                        return None;
                    }
                    (mask.index() & (1 << i) != 0) as u8
                }
                Cell::Empty => 0,
                Cell::Full => 1,
                Cell::Branch { .. } => return None,
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
                    return None;
                }
            }

            //  - The sign in the middle of a coarse face must agree with the
            //    sign of at least one of the face’s four corners.
            for i in 0..2 {
                let a: Corner<3> = (t * (i & 1 == 0)).into();
                let b = a | u;
                let c = a | v;
                let d = a | u | v;

                let center = cells[a.index()].corner(d);

                if [a, b, c, d]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return None;
                }
            }
            //  - The sign in the middle of a coarse cube must agree with the
            //    sign of at least one of the cube’s eight corners.
            for _i in 0..1 {
                // Doing this in the t,u,v loop isn't strictly necessary, but it
                // preserves the structure nicely.
                let center = cells[0].corner(t | u | v);
                if (0..8).all(|v| ((mask & (1 << v)) != 0) != center) {
                    return None;
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
        if CELL_TO_VERT_TO_EDGES[mask as usize].len() == 1 {
            Some(CellMask::new(mask))
        } else {
            None
        }
    }
}

impl std::ops::Index<CellIndex<3>> for Octree {
    type Output = Cell<3>;

    fn index(&self, i: CellIndex<3>) -> &Self::Output {
        match i.index {
            None => &self.root,
            Some((i, j)) => &self.cells[i][j as usize],
        }
    }
}

impl std::ops::IndexMut<CellIndex<3>> for Octree {
    fn index_mut(&mut self, i: CellIndex<3>) -> &mut Self::Output {
        match i.index {
            None => &mut self.root,
            Some((i, j)) => &mut self.cells[i][j as usize],
        }
    }
}

/// Data structure for an under-construction octree
#[derive(Debug)]
pub(crate) struct OctreeBuilder<F: Function + RenderHints> {
    /// In-construction octree
    pub(crate) octree: Octree,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,
}

impl<F: Function + RenderHints> Default for OctreeBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Function + RenderHints> OctreeBuilder<F> {
    /// Builds a new octree builder, which allocates data for 8 root cells
    pub(crate) fn new() -> Self {
        Self {
            octree: Octree::new(),
            eval_float_slice: Shape::<F>::new_float_slice_eval(),
            eval_grad_slice: Shape::<F>::new_grad_slice_eval(),
            eval_interval: Shape::<F>::new_interval_eval(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    /// Recurse down the octree, building the given cell
    ///
    /// Writes to `self.o.cells[cell]`, which must be reserved
    ///
    /// If a leaf is written, then `hermite` is populated
    fn recurse(
        &mut self,
        eval: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        cell: CellIndex<3>,
        max_depth: u8,
        hermite: &mut LeafHermiteData,
    ) {
        let (i, r) = self
            .eval_interval
            .eval_v(
                eval.i_tape(&mut self.tape_storage),
                cell.bounds[crate::mesh::types::X],
                cell.bounds[crate::mesh::types::Y],
                cell.bounds[crate::mesh::types::Z],
                vars,
            )
            .unwrap();
        self.octree[cell] = if i.upper() < 0.0 {
            Cell::Full
        } else if i.lower() > 0.0 {
            Cell::Empty
        } else {
            let sub_tape = if F::simplify_tree_during_meshing(cell.depth) {
                if let Some(trace) = r.as_ref() {
                    eval.simplify(
                        trace,
                        &mut self.workspace,
                        &mut self.shape_storage,
                        &mut self.tape_storage,
                    )
                } else {
                    eval
                }
            } else {
                eval
            };
            if cell.depth == max_depth as usize {
                self.leaf(sub_tape, vars, cell, hermite)
            } else {
                // Reserve new cells for the 8x children
                let index = self.octree.cells.len();
                self.octree.cells.push([Cell::Invalid; 8]);
                let mut hermite_child = [LeafHermiteData::default(); 8];
                for i in Corner::<3>::iter() {
                    let cell = cell.child(index, i);
                    self.recurse(
                        sub_tape,
                        vars,
                        cell,
                        max_depth,
                        &mut hermite_child[i.index()],
                    );
                }

                // Figure out whether the children can be collapsed
                self.octree.check_done(cell, index, hermite_child, hermite)
            }
        };
    }

    /// Evaluates the given leaf
    ///
    /// Writes the leaf vertex to `self.o.verts`, hermite data to
    /// `self.hermite`, and the leaf data to `self.leafs`.  Does **not** write
    /// anything to `self.o.cells`; the cell is returned instead.
    fn leaf(
        &mut self,
        eval: &mut RenderHandle<F>,
        vars: &ShapeVars<f32>,
        cell: CellIndex<3>,
        hermite_cell: &mut LeafHermiteData,
    ) -> Cell<3> {
        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::<3>::iter() {
            let [x, y, z] = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        let out = self
            .eval_float_slice
            .eval_v(eval.f_tape(&mut self.tape_storage), &xs, &ys, &zs, vars)
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

        let mask = CellMask::new(mask);

        // Start and endpoints in 3D space for intersection searches
        let mut start = [nalgebra::Vector3::zeros(); 12];
        let mut end = [nalgebra::Vector3::zeros(); 12];
        let mut edge_count = 0;

        // Convert from the corner mask to start and end-points in 3D space,
        // relative to the cell bounds (i.e. as a `Matrix3<u16>`).
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
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
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let ys =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let zs =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];

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
                .eval_v(eval.f_tape(&mut self.tape_storage), xs, ys, zs, vars)
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

        let xs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let ys = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let zs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];

        for (i, xyz) in intersections.iter().enumerate() {
            let pos = cell.pos(*xyz);
            xs[i] = Grad::new(pos.x, 1.0, 0.0, 0.0);
            ys[i] = Grad::new(pos.y, 0.0, 1.0, 0.0);
            zs[i] = Grad::new(pos.z, 0.0, 0.0, 1.0);
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grads = self
            .eval_grad_slice
            .eval_v(eval.g_tape(&mut self.tape_storage), xs, ys, zs, vars)
            .unwrap();

        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
            let mut force_point = None;
            let mut qef = QuadraticErrorSolver::new();
            for e in vs.iter() {
                let pos = nalgebra::Vector3::new(xs[i].v, ys[i].v, zs[i].v);
                let grad: nalgebra::Vector4<f32> = grads[i].into();

                // If a point has invalid gradients, then it's _probably_ on a
                // sharp feature in the mesh, so we should just snap to that
                // point specifically.  This means we don't solve the QEF, and
                // instead mark it as invalid.
                if grad.iter().any(|f| f.is_nan()) {
                    force_point = Some(pos);
                    hermite_cell.qef_err = QEF_ERR_INVALID;
                    break;
                }
                qef.add_intersection(pos, grad);

                // Record this intersection in the Hermite data for the leaf
                let edge_index = e.to_undirected().index();
                hermite_cell.intersections[edge_index] = LeafIntersection {
                    pos: nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0),
                    grad,
                };

                i += 1;
            }

            if let Some(pos) = force_point {
                verts.push(CellVertex { pos });
            } else {
                let (pos, err) = qef.solve();
                verts.push(pos);

                // We overwrite the error here, because it's only used when
                // collapsing cells, which only occurs if there's a single
                // vertex; last-error-wins works fine in that case.
                hermite_cell.qef_err = err;
            }
        }

        let index = self.octree.verts.len();
        self.octree.verts.extend(verts);
        self.octree.verts.extend(
            intersections
                .into_iter()
                .map(|pos| CellVertex { pos: cell.pos(pos) }),
        );

        Cell::Leaf(Leaf { mask, index })
    }
}

////////////////////////////////////////////////////////////////////////////////

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

    /// Error found when solving this QEF (if positive), or a special value
    qef_err: f32,
}

/// This QEF is not populated
const QEF_ERR_EMPTY: f32 = -1.0;

/// This QEF is known to be invalid and should be disregarded
const QEF_ERR_INVALID: f32 = -2.0;

impl Default for LeafHermiteData {
    fn default() -> Self {
        Self {
            intersections: Default::default(),
            face_qefs: Default::default(),
            center_qef: Default::default(),
            qef_err: QEF_ERR_EMPTY,
        }
    }
}

impl LeafHermiteData {
    /// Merges an octree subdivision of leaf hermite data
    ///
    /// Returns `None` if any of the leafs have invalid QEFs (typically due to
    /// NANs in normal computation).
    fn merge(leafs: [LeafHermiteData; 8]) -> Option<Self> {
        let mut out = Self::default();
        use super::types::{X, Y, Z};

        if leafs.iter().any(|v| v.qef_err == QEF_ERR_INVALID) {
            return None;
        }

        // Accumulate intersections along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();
            for edge in 0..4 {
                let mut start = Corner::<3>::new(0);
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
                let a = if face == 1 {
                    t.into()
                } else {
                    Corner::<3>::new(0)
                };
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
            let a = Corner::<3>::new(0);
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

        Some(out)
    }

    /// Solves the combined QEF
    pub fn solve(&self) -> (CellVertex<3>, f32) {
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
        context::Tree,
        mesh::types::{Edge, X, Y, Z},
        render::ThreadPool,
        render::View3,
        shape::EzShape,
        var::Var,
        vm::VmShape,
    };
    use nalgebra::Vector3;
    use std::collections::BTreeMap;

    fn depth0_single_thread() -> Settings<'static> {
        Settings {
            depth: 0,
            threads: None,
            ..Default::default()
        }
    }

    fn depth1_single_thread() -> Settings<'static> {
        Settings {
            depth: 1,
            threads: None,
            ..Default::default()
        }
    }

    fn sphere(center: [f32; 3], radius: f32) -> Tree {
        let (x, y, z) = Tree::axes();
        ((x - center[0]).square()
            + (y - center[1]).square()
            + (z - center[2]).square())
        .sqrt()
            - radius
    }

    fn cube(bx: [f32; 2], by: [f32; 2], bz: [f32; 2]) -> Tree {
        let (x, y, z) = Tree::axes();
        let x_bounds = (bx[0] - x.clone()).max(x - bx[1]);
        let y_bounds = (by[0] - y.clone()).max(y - by[1]);
        let z_bounds = (bz[0] - z.clone()).max(z - bz[1]);
        x_bounds.max(y_bounds).max(z_bounds)
    }

    #[test]
    fn test_cube_edge() {
        const EPSILON: f32 = 1e-3;
        let f = 2.0;
        let shape = VmShape::from(cube([-f, f], [-f, 0.3], [-f, 0.6]));
        // This should be a cube with a single edge running through the root
        // node of the octree, with an edge vertex at [0, 0.3, 0.6]
        let octree = Octree::build(&shape, depth0_single_thread());
        assert_eq!(octree.verts.len(), 5);
        let v = octree.verts[0].pos;
        let expected = nalgebra::Vector3::new(0.0, 0.3, 0.6);
        assert!(
            (v - expected).norm() < EPSILON,
            "bad edge vertex {v:?}; expected {expected:?}"
        );
    }

    fn cone(
        corner: nalgebra::Vector3<f32>,
        tip: nalgebra::Vector3<f32>,
        radius: f32,
    ) -> Tree {
        let dir = tip - corner;
        let length = dir.norm();
        let dir = dir.normalize();

        let corner = corner.map(|v| Tree::constant(v as f64));
        let dir = dir.map(|v| Tree::constant(v as f64));

        let (x, y, z) = Tree::axes();
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
        let shape = VmShape::from(sphere([0.0; 3], 0.2));

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&shape, depth0_single_thread());
        assert!(octree.cells.is_empty()); // root only
        assert_eq!(Cell::Empty, octree.root);
        assert!(octree.verts.is_empty());

        let empty_mesh = octree.walk_dual(depth0_single_thread());
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&shape, depth1_single_thread());
        assert_eq!(octree.cells.len(), 1); // one fanout of 8 cells
        assert_eq!(Cell::Branch { index: 0 }, octree.root);

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8, "incorrect vertex count");

        // Each cell is a leaf with 4 vertices (3 edges, 1 center)
        for o in octree.cells[0].iter() {
            let Cell::Leaf(Leaf { index, mask }) = *o else {
                panic!()
            };
            assert_eq!(mask.count_ones(), 1);
            assert_eq!(index % 4, 0);
        }

        let sphere_mesh = octree.walk_dual(depth1_single_thread());
        assert!(sphere_mesh.vertices.len() > 1);
        assert!(!sphere_mesh.triangles.is_empty());
    }

    #[test]
    fn test_sphere_verts() {
        let shape = VmShape::from(sphere([0.0; 3], 0.2));

        let octree = Octree::build(&shape, depth1_single_thread());
        let sphere_mesh = octree.walk_dual(depth1_single_thread());

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
        let shape = VmShape::from(sphere([0.0; 3], 0.85));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, settings);
            let sphere_mesh = octree.walk_dual(settings);

            check_for_vertex_dupes(&sphere_mesh).unwrap();
            check_for_edge_matching(&sphere_mesh).unwrap();
        }
    }

    #[test]
    fn test_cube_verts() {
        let shape = VmShape::from(cube([-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]));

        let octree = Octree::build(&shape, depth1_single_thread());
        let mesh = octree.walk_dual(depth1_single_thread());
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
        for dx in [0.0, 0.25, -0.25, 2.0, -2.0] {
            for dy in [0.0, 0.25, -0.25, 2.0, -2.0] {
                for offset in [0.0, -0.2, 0.2] {
                    let (x, y, z) = Tree::axes();
                    let f = x * dx + y * dy + z + offset;
                    let shape = VmShape::from(f);
                    let octree = Octree::build(&shape, depth0_single_thread());

                    assert!(octree.cells.is_empty()); // root only
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
                        let (r, _) = eval.eval(&tape, v.x, v.y, v.z).unwrap();
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
            let corner = nalgebra::Vector3::new(-1.0, -1.0, -1.0);
            let shape = VmShape::from(cone(corner, tip, 0.1));

            let mut eval = VmShape::new_point_eval();
            let tape = shape.ez_point_tape();
            let (v, _) = eval.eval(&tape, tip.x, tip.y, tip.z).unwrap();
            assert!(v.abs() < 1e-6, "bad tip value: {v}");
            let (v, _) =
                eval.eval(&tape, corner.x, corner.y, corner.z).unwrap();
            assert!(v < 0.0, "bad corner value: {v}");

            let octree = Octree::build(&shape, depth0_single_thread());
            assert!(octree.cells.is_empty()); // root only
            assert_eq!(octree.verts.len(), 4);

            let pos = octree.verts[0].pos;
            assert!(
                (pos - tip).norm() < 1e-3,
                "bad vertex position: expected {tip:?}, got {pos:?}"
            );
        }
    }

    fn test_mesh_manifold_inner(threads: Option<&ThreadPool>, mask: u8) {
        let mut shape = vec![];
        for j in Corner::<3>::iter() {
            if mask & (1 << j.index()) != 0 {
                shape.push(sphere(
                    [
                        if j & X { 0.5 } else { 0.0 },
                        if j & Y { 0.5 } else { 0.0 },
                        if j & Z { 0.5 } else { 0.0 },
                    ],
                    0.1,
                ));
            }
        }
        let Some(start) = shape.pop() else { return };
        let shape = shape.into_iter().fold(start, |acc, s| acc.min(s));

        // Now, we have our shape, which is 0-8 spheres placed at the
        // corners of the cell spanning [0, 0.25]
        let shape = VmShape::from(shape);
        let settings = Settings {
            depth: 2,
            threads,
            ..Default::default()
        };
        let octree = Octree::build(&shape, settings);

        let mesh = octree.walk_dual(settings);
        if mask != 0 && mask != 255 {
            assert!(!mesh.vertices.is_empty());
            assert!(!mesh.triangles.is_empty());
        }

        if let Err(e) = check_for_vertex_dupes(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
        if let Err(e) = check_for_edge_matching(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
    }

    #[test]
    fn test_mesh_manifold_single_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(None, mask)
        }
    }

    #[test]
    fn test_mesh_manifold_multi_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(Some(&ThreadPool::Global), mask)
        }
    }

    #[test]
    fn test_collapsible() {
        let shape = VmShape::from(sphere([0.0; 3], 0.1));
        let octree = Octree::build(&shape, depth1_single_thread());
        assert!(octree.collapsible(0).is_none());

        // If we have a single corner sphere, then the builder will collapse the
        // branch for us, leaving just a leaf.
        let shape = VmShape::from(sphere([-1.0; 3], 0.1));
        let octree = Octree::build(&shape, depth1_single_thread());
        assert!(matches!(octree.root, Cell::Leaf { .. }));

        // Test with a single thread and a deeper tree, which should still work
        let octree = Octree::build(
            &shape,
            Settings {
                depth: 4,
                threads: None,
                ..Default::default()
            },
        );
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        // This should also work with multiple threads!
        let octree = Octree::build(
            &shape,
            Settings {
                depth: 4,
                threads: Some(&ThreadPool::Global),
                ..Default::default()
            },
        );
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        let shape = VmShape::from(sphere([-1.0, 0.0, 1.0], 0.1));
        let octree = Octree::build(&shape, depth1_single_thread());
        assert!(octree.collapsible(0).is_none());

        let a = sphere([-1.0; 3], 0.1);
        let b = sphere([1.0; 3], 0.1);
        let shape = VmShape::from(a.min(b));
        let octree = Octree::build(&shape, depth1_single_thread());
        assert!(octree.collapsible(0).is_none());
    }

    #[test]
    fn test_empty_collapse() {
        // Make a very smol sphere that won't be sampled
        let shape = VmShape::from(sphere([0.1; 3], 0.05));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 1,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, settings);
            assert_eq!(
                octree.root,
                Cell::Empty,
                "failed to collapse octree with threads: {:?}",
                threads.map(|t| t.thread_count())
            );
        }
    }

    #[test]
    fn test_colonnade_manifold() {
        const COLONNADE: &str = include_str!("../../../models/colonnade.vm");
        let (ctx, root) =
            crate::Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, settings);
            let mesh = octree.walk_dual(settings);
            // Note: the model has duplicate vertices!
            if let Err(e) = check_for_edge_matching(&mesh) {
                panic!(
                    "colonnade model has {e} with threads: {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn colonnade_bounds() {
        const COLONNADE: &str = include_str!("../../../models/colonnade.vm");
        let (ctx, root) =
            crate::Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 8,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, settings);
            let mesh = octree.walk_dual(settings);
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -1.0
                        && v.y < 1.0
                        && v.y > -1.0
                        && v.z < 1.0
                        && v.z > -0.5,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn bear_bounds() {
        const COLONNADE: &str = include_str!("../../../models/bear.vm");
        let (ctx, root) =
            crate::Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, settings);
            let mesh = octree.walk_dual(settings);
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -0.75
                        && v.y < 1.0
                        && v.y > -0.75
                        && v.z < 0.75
                        && v.z > -0.75,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
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
        let mut hermite = LeafHermiteData::default();

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

        let merged = LeafHermiteData::merge(hermites).unwrap();
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
        let shape = VmShape::from(sphere([0.0; 3], 0.75));

        let settings = Settings {
            depth: 4,
            ..Default::default()
        };

        let octree = Octree::build(&shape, settings).walk_dual(settings);
        for v in octree.vertices.iter() {
            let n = v.norm();
            assert!(n > 0.7 && n < 0.8, "invalid vertex at {v:?}: {n}");
        }
    }

    #[test]
    fn test_octree_camera() {
        let shape = VmShape::from(sphere([1.0; 3], 0.25));

        let center = Vector3::new(1.0, 1.0, 1.0);
        let settings = Settings {
            depth: 4,
            view: View3::from_center_and_scale(center, 0.5),
            threads: None,
        };

        let octree = Octree::build(&shape, settings).walk_dual(settings);
        for v in octree.vertices.iter() {
            let n = (v - center).norm();
            assert!(n > 0.2 && n < 0.3, "invalid vertex at {v:?}: {n}");
        }
    }

    #[test]
    fn test_mesh_vars() {
        let (x, y, z) = Tree::axes();
        let v = Var::new();
        let c = Tree::from(v);
        let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
        let shape = VmShape::from(sphere);

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 4,
                threads,
                view: View3::default(),
            };

            for r in [0.5, 0.75] {
                let mut vars = ShapeVars::new();
                vars.insert(v.index().unwrap(), r);
                let octree = Octree::build_with_vars(&shape, &vars, settings)
                    .walk_dual(settings);
                for v in octree.vertices.iter() {
                    let n = v.norm();
                    assert!(
                        n > r - 0.05 && n < r + 0.05,
                        "invalid vertex at {v:?}: {n} != {r} with threads {:?}",
                        threads.map(|t| t.thread_count())
                    );
                }
            }
        }
    }
}
