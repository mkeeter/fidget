//! Post-generation cleanup of octrees for manifold dual contouring

use super::{
    cell::{Cell, CellIndex, CellVertex, Leaf},
    dc::{self, DcBuilder},
    frame::Frame,
    gen::CELL_TO_VERT_TO_EDGES,
    types::{Edge, Face},
    types::{X, Y, Z},
    Octree, Settings,
};

/// Overload dual contouring's tree walk to mark leafs that need subdivision
pub struct DcFixup {
    pub needs_fixing: Vec<bool>,
    max_depth: usize,
    verts: Vec<(nalgebra::Vector3<f32>, CellIndex)>,
}

impl DcFixup {
    pub fn new(size: usize, settings: &Settings) -> Self {
        Self {
            needs_fixing: vec![false; size],
            max_depth: settings.max_depth as usize,
            verts: vec![],
        }
    }
    pub fn mark(&mut self, cell: CellIndex) {
        if cell.depth < self.max_depth {
            self.needs_fixing[cell.index] = true;
        }
    }
}

impl DcBuilder for DcFixup {
    type VertexIndex = usize;

    fn cell(&mut self, octree: &Octree, cell: CellIndex) {
        if let Cell::Leaf(Leaf { index, mask }) =
            octree.cells[cell.index].into()
        {
            for i in 0..CELL_TO_VERT_TO_EDGES[mask as usize].len() {
                if !cell.bounds.contains(octree.verts[index + i]) {
                    self.mark(cell);
                }
            }
        }
        dc::dc_cell(octree, cell, self);
    }

    fn face<F: Frame>(&mut self, octree: &Octree, a: CellIndex, b: CellIndex) {
        if a.depth == b.depth && (octree.is_leaf(a) || octree.is_leaf(b)) {
            let mut common = None;
            for axis in [X, Y, Z] {
                if a.bounds[axis].upper() == b.bounds[axis].lower() {
                    assert!(common.is_none());
                    common = Some((axis, 1))
                }
                if a.bounds[axis].lower() == b.bounds[axis].upper() {
                    assert!(common.is_none());
                    common = Some((axis, 0))
                }
            }
            let Some((axis, v)) = common else {
                panic!("faces do not touch")
            };
            let fa = Face::new((axis.index() * 2 + v).try_into().unwrap());
            let fb = Face::new((axis.index() * 2 + 1 - v).try_into().unwrap());

            let ma = octree.face_mask(a, fa);
            let mb = octree.face_mask(b, fb);

            if ma.is_none() {
                assert!(mb.is_some());
                self.mark(b);
            }
            if mb.is_none() {
                assert!(ma.is_some());
                self.mark(a);
            }
        }
        // ...and recurse
        dc::dc_face::<F, DcFixup>(octree, a, b, self);
    }

    fn edge<F: Frame>(
        &mut self,
        octree: &Octree,
        a: CellIndex,
        b: CellIndex,
        c: CellIndex,
        d: CellIndex,
    ) {
        let cs = [a, b, c, d];
        #[allow(clippy::identity_op)]
        if cs.iter().all(|v| v.depth == a.depth)
            && cs.iter().any(|v| octree.is_leaf(*v))
        {
            let (t, _u, _v) = F::frame();
            let e = 4 * t.index();
            let ea =
                octree.edge_mask(a, Edge::new((e + 3).try_into().unwrap()));
            let eb =
                octree.edge_mask(b, Edge::new((e + 2).try_into().unwrap()));
            let ec =
                octree.edge_mask(c, Edge::new((e + 0).try_into().unwrap()));
            let ed =
                octree.edge_mask(d, Edge::new((e + 1).try_into().unwrap()));

            let edge_masks = [ea, eb, ec, ed];
            if edge_masks.iter().any(|c| c.is_none()) {
                for (e, v) in edge_masks.iter().zip(cs.iter()) {
                    if e.is_some() {
                        self.mark(*v);
                    }
                }
            }
        }
        dc::dc_edge::<F, DcFixup>(octree, a, b, c, d, self);
    }

    fn triangle(&mut self, a: usize, b: usize, _c: usize) {
        let (va, ca) = self.verts[a];
        let (vb, cb) = self.verts[b];

        // Pick the face which should be intersected by the edge, and the value
        // at that shared face.
        // TODO: should we pass a Frame parameter to this function instead?
        let mut common = None;
        for axis in [X, Y, Z] {
            if ca.bounds[axis].upper() == cb.bounds[axis].lower() {
                assert!(common.is_none());
                common = Some((axis, ca.bounds[axis].upper()))
            }
            if ca.bounds[axis].lower() == cb.bounds[axis].upper() {
                assert!(common.is_none());
                common = Some((axis, ca.bounds[axis].lower()))
            }
        }
        let Some((axis, v)) = common else {
            panic!("faces do not touch")
        };

        let dist = (v - va[axis.index()]) / (vb - va).normalize()[axis.index()];
        let hit = va + dist * (vb - va).normalize();
        // TODO: what if va ≈ vb?

        // We check against the bounds of the deeper cell, which is physically
        // smaller and has a numerically higher `depth`
        let bounds = if ca.depth > cb.depth {
            ca.bounds
        } else {
            cb.bounds
        };

        let u = axis.next();
        let v = u.next();

        // Check if the va-vb line is bounded by the cell boundaries at the
        // point where it crosses the face.
        if !bounds[u].contains(hit[u.index()])
            || !bounds[v].contains(hit[v.index()])
        {
            use std::cmp::Ordering;
            // We tag the larger cell for fixup, which has a numerically smaller
            // `depth`.  If both cells are at the same depth, then we tag both,
            // unless one or the other has already been tagged for other reasons
            match ca.depth.cmp(&cb.depth) {
                Ordering::Greater => {
                    if !self.needs_fixing[ca.index] {
                        self.mark(cb);
                    }
                }
                Ordering::Less => {
                    if !self.needs_fixing[cb.index] {
                        self.mark(ca);
                    }
                }
                Ordering::Equal => {
                    if !self.needs_fixing[ca.index]
                        && !self.needs_fixing[cb.index]
                    {
                        self.mark(ca);
                        self.mark(cb);
                    }
                }
            }
        }
    }

    fn vertex(
        &mut self,
        v: usize,
        cell: CellIndex,
        verts: &[CellVertex],
    ) -> usize {
        let next_vert = self.verts.len();
        self.verts.push((verts[v].pos, cell));
        next_vert
    }

    fn invalid_leaf_vert(&mut self, a: CellIndex) {
        self.mark(a);
    }

    fn fan_done(&mut self) {
        self.verts.clear();
    }
}
