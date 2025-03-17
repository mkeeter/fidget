//! Dual contouring implementation

use crate::mesh::{
    builder::MeshBuilder,
    cell::{Cell, CellIndex},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{Corner, Edge, X, Y, Z},
    Octree,
};

pub fn dc_cell(octree: &Octree, cell: CellIndex<3>, out: &mut MeshBuilder) {
    if let Cell::Branch { index } = octree[cell] {
        debug_assert_eq!(index % 8, 0);
        for i in Corner::iter() {
            out.cell(octree, octree.child(cell, i));
        }

        // Helper function for DC face calls
        fn dc_faces<T: Frame>(
            octree: &Octree,
            cell: CellIndex<3>,
            out: &mut MeshBuilder,
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
        dc_faces::<XYZ>(octree, cell, out);
        dc_faces::<YZX>(octree, cell, out);
        dc_faces::<ZXY>(octree, cell, out);

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
pub fn dc_face<T: Frame>(
    octree: &Octree,
    lo: CellIndex<3>,
    hi: CellIndex<3>,
    out: &mut MeshBuilder,
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
pub fn dc_edge<T: Frame>(
    octree: &Octree,
    a: CellIndex<3>,
    b: CellIndex<3>,
    c: CellIndex<3>,
    d: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    let cs = [a, b, c, d];
    if cs.iter().all(|v| octree.is_leaf(*v)) {
        // If any of the leafs are Empty or Full, then this edge can't
        // include a sign change.  TODO: can we make this any -> all if we
        // collapse empty / filled leafs into Empty / Full cells?
        let leafs = cs.map(|cell| match octree[cell] {
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
            let start = !(leafs[deepest].mask & start);
            let end = !(leafs[deepest].mask & end);
            // If there is no sign change, then there's nothing to do here.
            if start == end {
                return;
            }
            start
        };

        // Iterate over each of the edges, assigning a vertex if the sign change
        // lines up.
        let mut verts = [None; 4];
        for i in 0..4 {
            if cs[i].depth == cs[deepest].depth {
                let (start, end) = edges[i].corners();
                let s = !(leafs[i].mask & start);
                let e = !(leafs[i].mask & end);
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
                    panic!("invalid leaf vertex at {a:?}");
                }
            }
        }

        let verts = verts.map(Option::unwrap);

        // Pick the intersection vertex based on the deepest cell
        let i = out.vertex(
            leafs[deepest].index + verts[deepest].edge.0 as usize,
            &octree.verts,
        );
        // Helper function to extract other vertices
        let mut vert = |i: usize| {
            out.vertex(leafs[i].index + verts[i].vert.0 as usize, &octree.verts)
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
