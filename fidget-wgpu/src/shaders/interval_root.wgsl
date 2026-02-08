// Interval root tile evaluation
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

@group(1) @binding(0) var<storage, read_write> tape_data: TapeData;

/// Tiles which need to be subdivided and evaluated
@group(1) @binding(1) var<storage, read_write> tile64_out: TileListOutput;

// Array of filled tiles, laid out in X/Y order
@group(1) @binding(2) var<storage, read_write> tile64_zmin: array<Voxel>;

// Number of tiles for each Z value (used for a quasi-radix sort)
@group(1) @binding(3) var<storage, read_write> tile64_hist: array<atomic<u32>>;

/// Root tile size
const TILE_SIZE: u32 = 64;

@compute @workgroup_size(4, 4, 4)
fn interval_root_main(
    @builtin(global_invocation_id) tile_corner: vec3u
) {
    // Calculate render size in tile units
    let size_tiles = config.render_size / TILE_SIZE;

    if (tile_corner.x >= size_tiles.x ||
        tile_corner.y >= size_tiles.y ||
        tile_corner.z >= size_tiles.z)
    {
        return;
    }

    // Tile's lower z position, in voxels
    let corner_pos = tile_corner * TILE_SIZE;

    // Compute transformed interval regions
    let m = interval_inputs(tile_corner, TILE_SIZE);

    // Do the actual interpreter work
    var stack = Stack();
    let out = run_tape(0u, m, &stack);
    let v = out.value.v;

    // If the tile is completely empty, then we're done!
    if v[0] > 0.0 {
        return;
    }

    // We'll simplify the tape for both full and ambiguous tiles, because we'll
    // want a simplified tape for normal rendering later on.
    let next = simplify_tape(out.pos, out.count, &stack);

    if v[1] < 0.0 {
        let tile_index_xy = tile_corner.x + tile_corner.y * size_tiles.x;
        let new_z = corner_pos.z + TILE_SIZE - 1;
        let new_value = (new_z << 20) | next; // TODO next overflow
        atomicMax(&tile64_zmin[tile_index_xy].value, new_value);
    } else {
        // Otherwise, enqueue the tile and add its Z position to the histogram
        let tile_index_xyz = tile_corner.x +
            tile_corner.y * size_tiles.x +
            tile_corner.z * size_tiles.x * size_tiles.y;
        let offset = atomicAdd(&tile64_out.count, 1u);
        tile64_out.tiles[offset] = ActiveTile(tile_index_xyz, next);
        atomicAdd(&tile64_hist[tile_corner.z], 1);
    }
}
