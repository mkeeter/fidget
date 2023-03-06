use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    fmt::Write,
};

const X: usize = 1;
const Y: usize = 2;
const Z: usize = 4;

fn main() {
    if std::env::var("CARGO_FEATURE_MESH").is_ok() {
        build_mdc_table();
    }
}

/// Builds a table for Manifold Dual Contouring connectivity.
///
/// This is roughly equivalent to Figure 5 in Nielson's Dual Marching Cubes
/// (2004), but worked out automatically by clustering cell corners.
fn build_mdc_table() {
    let mut vert_table =
        "const CELL_TO_VERTS: [&'static [&'static [(u8, u8)]]; 256] = [\n"
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
        // vert_map is a map from (start region, end region) to a vertex index.
        // verts is a map from vertex index to the edges that built that vertex.
        let mut vert_map = BTreeMap::new();
        let mut verts = vec![];
        // TODO: just store edges in `vert_map` instead?
        for start in 0..8 {
            for axis in [X, Y, Z] {
                let end = start ^ axis;
                // We're only looking for inside (1) -> outside (0) transitions
                // here, and will skip everything else.
                if (i & (1 << start)) != 0 && (i & (1 << end)) == 0 {
                    let start_region = regions[start];
                    let end_region = regions[end];
                    assert!(start_region != end_region);
                    if let Entry::Vacant(e) = vert_map.entry(start_region) {
                        e.insert(verts.len());
                        verts.push(BTreeSet::new());
                    }
                    verts[vert_map[&start_region]].insert((start, end));
                }
            }
        }
        // There are two maps associated with this cell:
        // - A list of vertices, each of which has a list of transition edges
        // - A map from transition edge to vertex in the previous list
        _ = writeln!(&mut vert_table, "&[");
        for edges in &verts {
            _ = write!(&mut vert_table, "    &[");
            for (start, end) in edges {
                _ = write!(&mut vert_table, "({start}, {end}), ");
            }
            _ = writeln!(&mut vert_table, "],");
        }
        _ = writeln!(&mut vert_table, "],");
    }
    _ = writeln!(&mut vert_table, "];");

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("mdc_tables.rs");
    std::fs::write(dest_path, vert_table).unwrap();
}
