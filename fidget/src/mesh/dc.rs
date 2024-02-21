//! Dual contouring implementation

use crate::mesh::{
    cell::{Cell, CellIndex, CellVertex},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{Corner, Edge, X, Y, Z},
    Octree,
};

pub trait DcBuilder {
    /// Type for vertex indexes
    ///
    /// This is typically a `usize`, but we'll sometimes explicitly force it to
    /// be a `u64` if we're planning to use upper bits for flags.
    type VertexIndex: Copy + Clone;

    fn cell(&mut self, octree: &Octree, cell: CellIndex);
    fn face<F: Frame>(&mut self, octree: &Octree, a: CellIndex, b: CellIndex);

    /// Handles four cells that share a common edge aligned on axis `T`
    ///
    /// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
    /// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
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
    fn triangle(
        &mut self,
        a: Self::VertexIndex,
        b: Self::VertexIndex,
        c: Self::VertexIndex,
    );

    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`](super::Octree::verts).
    fn vertex(
        &mut self,
        v: usize,
        cell: CellIndex,
        verts: &[CellVertex],
    ) -> Self::VertexIndex;
}

pub fn dc_cell<B: DcBuilder>(octree: &Octree, cell: CellIndex, out: &mut B) {
    if let Cell::Branch { index, .. } = octree[cell].into() {
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
        let leafs = cs.map(|cell| match octree[cell].into() {
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
