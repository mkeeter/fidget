//! Post-generation cleanup of octrees for manifold dual contouring

use super::{
    cell::{Cell, CellIndex, CellVertex, Leaf},
    dc::{self, DcBuilder},
    frame::Frame,
    gen::CELL_TO_VERT_TO_EDGES,
    types::{Edge, Face},
    types::{X, Y, Z},
    Octree,
};

/// Overload dual contouring's tree walk to mark leafs that need subdivision
pub struct DcFixup {
    pub needs_fixing: Vec<bool>,
    verts: Vec<(nalgebra::Vector3<f32>, CellIndex)>,
}

impl DcFixup {
    pub fn new(size: usize) -> Self {
        Self {
            needs_fixing: vec![false; size],
            verts: vec![],
        }
    }
}

impl DcBuilder for DcFixup {
    fn cell(&mut self, octree: &Octree, cell: CellIndex) {
        if let Cell::Leaf(Leaf { index, mask }) =
            octree.cells[cell.index].into()
        {
            for i in 0..CELL_TO_VERT_TO_EDGES[mask as usize].len() {
                if !octree.verts[index + i].valid() {
                    self.needs_fixing[cell.index] = true;
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
            let Some((axis, v)) = common else { panic!("faces do not touch") };
            let fa = Face::new((axis.index() * 2 + v).try_into().unwrap());
            let fb = Face::new((axis.index() * 2 + 1 - v).try_into().unwrap());

            let ma = octree.face_mask(a, fa);
            let mb = octree.face_mask(b, fb);

            // For now, we know that face masks are always equal, because we
            // aren't doing any fixing up (which could introduce non-manifold
            // cells next to each other).
            assert_eq!(ma, mb);

            if ma.is_none() {
                self.needs_fixing[a.index] = true;
            }
            if mb.is_none() {
                self.needs_fixing[b.index] = true;
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

            // This won't always be true!
            assert_eq!(ea, eb);
            assert_eq!(ea, ec);
            assert_eq!(ea, ed);

            if ea.is_none() {
                self.needs_fixing[a.index] = true;
            }
            if eb.is_none() {
                self.needs_fixing[b.index] = true;
            }
            if ec.is_none() {
                self.needs_fixing[c.index] = true;
            }
            if ed.is_none() {
                self.needs_fixing[d.index] = true;
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
        let Some((axis, v)) = common else { panic!("faces do not touch") };

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
            // `depth`.  If both cells are at the same depth, then we tag both!
            match ca.depth.cmp(&cb.depth) {
                Ordering::Greater => self.needs_fixing[cb.index] = true,
                Ordering::Less => self.needs_fixing[ca.index] = true,
                Ordering::Equal => {
                    self.needs_fixing[ca.index] = true;
                    self.needs_fixing[cb.index] = true;
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
        self.verts.push((cell.pos(verts[v]), cell));
        next_vert
    }

    fn invalid_leaf_vert(&mut self, a: CellIndex) {
        self.needs_fixing[a.index] = true;
    }

    fn fan_done(&mut self) {
        self.verts.clear();
    }
}
