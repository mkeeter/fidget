// Interval root tile evaluation
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

@group(1) @binding(0) var<storage, read_write> tiles_out: TileListOutput;
@group(1) @binding(1) var<storage, read_write> tile64_zmax: array<atomic<u32>>;

/// Root tile size
const TILE_SIZE: u32 = 64;

@compute @workgroup_size(4, 4, 4)
fn interval_root_main(
    @builtin(global_invocation_id) tile_corner: vec3u
) {
    // Calculate render size in tile units
    let size64 = config.render_size / 64;

    if (tile_corner.x >= size64.x ||
        tile_corner.y >= size64.y ||
        tile_corner.z >= size64.z)
    {
        return;
    }

    let tile_index_xy = tile_corner.x + tile_corner.y * size64.x;
    let tile_index_xyz = tile_index_xy + tile_corner.z * size64.x * size64.y;

    // Tile's lower z position, in voxels
    let corner_pos = tile_corner * TILE_SIZE;
    let corner_z = corner_pos.z;

    // Compute transformed interval regions
    let m = interval_inputs(tile_corner, TILE_SIZE);

    // Do the actual interpreter work
    var stack = Stack();
    let out = run_tape(0u, m, &stack);
    let v = out.value.v;

    // If the tile is completely empty, then we're done; we've already written
    // the tile tape to 0u, and there's nothing else to do.
    if v[0] > 0.0 {
        return;
    }

    // The tile is full, so set the "filled" flag when pushing the tile to the
    // tape list, which short-circuits evaluation.  We do this instead of just
    // setting tile_zmin so that the tile is evaluated when rendering normals,
    // because we need to compute normals for filled pixels in root tiles.
    var filled_bit = 0u;
    if v[1] < 0.0 {
        filled_bit = 1 << 31u;
    }

    // We have to subdivide and recurse, which we do by writing the 64^3
    // tile and incrementing our dispatch size (in a particular strata).

    // Write the tile into the output list (not yet packed into strata)
    let offset = atomicAdd(&tiles_out.count, 1u);
    tiles_out.active_tiles[offset] = tile_index_xyz | filled_bit;

    // Store the max Z tile for each XY position, so we can pack into strata
    atomicMax(&tile64_zmax[tile_index_xy], tile_corner.z);

    let next = simplify_tape(out.pos, out.count, &stack);
    if next != 0 {
        // Update this tile's position in the tape index map
        let tape_index = get_tape_offset_for_level(corner_pos, 64u);
        tile_tape[tape_index] = next;
    }
}

/// Allocates a new chunk, returning a past-the-end pointer
///
/// Note that we increment both the current tape data offset *and* the root tape
/// length, because we want to preserve these tapes through the reset in
/// `backfill` (which resets by assigning `tape_data_offset = root_tape_len`).
/// There's no concern about racing multiple simultaneous calls to `alloc`,
/// because the `root_tape_len` is the canonical offset.
fn alloc(chunk_size: u32) -> u32 {
    atomicAdd(&config.tape_data_offset, chunk_size);
    return atomicAdd(&config.root_tape_len, chunk_size);
}

/// Undo an allocation
fn dealloc(chunk_size: u32) {
    atomicSub(&config.tape_data_offset, chunk_size);
    atomicSub(&config.root_tape_len, chunk_size);
}
