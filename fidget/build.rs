use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
};

// Same axes as in `fidget::mesh`, but available at build time.
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 4;

fn next(axis: usize) -> usize {
    match axis {
        X => Y,
        Y => Z,
        Z => X,
        a => panic!("Invalid axis {a}"),
    }
}

fn main() {
    // Check CPU feature support and error out if we don't have the appropriate
    // features. This isn't a fool-proof – someone could build on a machine with
    // AVX2 support, then try running those binaries elsewhere – but is a good
    // first line of defense.
    if std::env::var("CARGO_FEATURE_JIT").is_ok() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!(
                "`x86_64` build with `jit` enabled requires AVX2 instructions"
            );
            std::process::exit(1);
        }

        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!(
                "`aarch64` build with `jit` enabled requires NEON instructions"
            );
            std::process::exit(1);
        }
    }

    if std::env::var("CARGO_FEATURE_MESH").is_ok() {
        build_mdc_table();
    }
}

/// Builds a table for Manifold Dual Contouring connectivity.
///
/// This is roughly equivalent to Figure 5 in Nielson's Dual Marching Cubes
/// (2004), but worked out automatically by clustering cell corners.
fn build_mdc_table() {
    let mut vert_table = "
/// Lookup table to find edges for a particular cell configuration
///
/// Given a cell index `i` (as an 8-bit value), looks up a list of vertices
/// which are required for that cell.  Each vertex is implicitly numbered based
/// on its position in the list, and itself stores a list of edges (as packed
/// directed values in the range 0-24).
pub const CELL_TO_VERT_TO_EDGES: [&'static [&'static [u8]]; 256] = [\n"
        .to_owned();

    // TODO: refactor this to `CELL_TO_VERT_TO_INTERSECTION`, which returns an
    // implicitly numbered intersection, and `CELL_TO_INTERSECTION_TO_EDGE`
    // which looks up that intersection and returns `(start, end)`
    //
    // Then, CELL_TO_EDGE_TO_VERT could be only [[u8; 12; 256], since we
    // wouldn't care about edge signs.

    let mut edge_table = "
/// Lookup table to find which vertex is associated with a particular edge
///
/// Given a cell index `i` (as an 8-bit value) and an edge index `e` (as a
/// packed undirected value in the range 0-12), returns a vertex index `v` such
/// that [`CELL_TO_VERT_TO_EDGES[i][v]`](CELL_TO_VERT_TO_EDGES) contains the
/// edge `e` (as a directed edge, i.e. possibly shifted by 12)
pub const CELL_TO_EDGE_TO_VERT: [[u8; 12]; 256] = [\n"
        .to_owned();
    for i in 0..256 {
        let mut filled_regions = BTreeMap::new();
        let mut empty_regions = BTreeMap::new();
        for j in 0..8 {
            if (i & (1 << j)) == 0 {
                empty_regions.insert(j, 1 << j);
            } else {
                filled_regions.insert(j, 1 << j);
            }
        }
        // Collapse connected cells in both filled and empty regions
        for r in [&mut filled_regions, &mut empty_regions] {
            loop {
                let mut changed = false;
                let mut next = r.clone();
                for f in r.keys() {
                    for axis in [X, Y, Z] {
                        let g = &(f ^ axis);
                        if r.contains_key(g) {
                            let v = next[f] | next[g];
                            changed |= (next[f] != v) | (next[g] != v);

                            *next.get_mut(f).unwrap() = v;
                            *next.get_mut(g).unwrap() = v;
                        }
                    }
                }
                *r = next;
                if !changed {
                    break;
                }
            }
        }
        // At this point, {filled,empty}_regions are maps from a vertex
        // number (0-7) to a mask of the region containing that vertex.
        //
        // We can discard the vertex numbers and just store the region masks
        // before processing them further
        let filled_regions: BTreeSet<u8> =
            filled_regions.into_values().collect();
        let empty_regions: BTreeSet<u8> = empty_regions.into_values().collect();

        // Now, we can flatten into a map from vertex (0-7) to an abstract
        // region number (0-), since that's what actually matters when
        // grouping transitions.
        let mut regions = [u8::MAX; 8];
        for (i, r) in filled_regions
            .into_iter()
            .chain(empty_regions.into_iter())
            .enumerate()
        {
            for j in 0..8 {
                if r & (1 << j) != 0 {
                    assert_eq!(regions[j], u8::MAX);
                    regions[j as usize] = i as u8;
                }
            }
        }

        // We're finally ready to build the edge transition table!
        //
        // vert_map is a map from start region to a vertex, defined as a list
        // of edges that built that vertex.
        let mut verts: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut edge = 0;
        for rev in [false, true] {
            for t in [X, Y, Z] {
                // t-u-v forms a right-handed coordinate system
                let u = next(t);
                let v = next(u);
                for b in 0..2 {
                    for a in 0..2 {
                        let start = (a * u) | (b * v);
                        let end = start | t;

                        let (start, end) =
                            if rev { (end, start) } else { (start, end) };
                        println!("checking {start} -> {end}");

                        // Only process this edge if `start` is inside the model
                        // (non-zero) and `end` is outside (0).
                        if ((i & (1 << start)) != 0) && ((i & (1 << end)) == 0)
                        {
                            let start_region = regions[start];
                            println!(
                                "   got hit in {start_region}, {t} {u} {v}"
                            );
                            let end_region = regions[end];
                            assert!(start_region != end_region);
                            verts.entry(start_region).or_default().push(edge);
                        }
                        edge += 1;
                    }
                }
            }
        }
        println!("verts: {verts:?}");

        // There are two maps associated with this cell:
        // - A list of vertices, each of which has a list of transition edges
        // - A map from transition edge to vertex in the previous list
        let mut edge_map = [u8::MAX; 12];
        writeln!(&mut vert_table, "  &[").unwrap();
        for (vert, (_, edges)) in verts.iter().enumerate() {
            write!(&mut vert_table, "    &[").unwrap();
            for e in edges {
                let rev = e / 12 != 0;

                let t = 1 << ((e % 12) / 4);
                let u = next(t);
                let v = next(u);

                let a = (u * (e % 2)) | (v * ((e % 4) / 2));
                let b = a | t;

                let (start, end) = if rev { (b, a) } else { (a, b) };

                assert!((i & (1 << start)) != 0);
                assert!((i & (1 << end)) == 0);

                write!(&mut vert_table, "{e}, ").unwrap();

                // Convert from directed to undirected edge for the second map
                edge_map[*e % 12] = vert.try_into().unwrap();
            }
            writeln!(&mut vert_table, "],").unwrap();
        }
        writeln!(&mut vert_table, "  ],").unwrap();
        writeln!(&mut edge_table, "    {edge_map:?},").unwrap();
    }
    writeln!(&mut vert_table, "];").unwrap();
    writeln!(&mut edge_table, "];").unwrap();

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("mdc_tables.rs");
    std::fs::write(dest_path, format!("{}\n{}", vert_table, edge_table))
        .unwrap();
}
