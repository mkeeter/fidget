// Interval root tile evaluation
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`


// This is a set of per-strata `TileListOutput` arrays.  Each one is
// `strata_size_bytes(..)` long, which is large enough to fit every tile.  We
// can't represent this directly, so good luck poking the right memory locations
// by hand!
@group(1) @binding(0) var<storage, read_write> tiles_out: array<atomic<u32>>;

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

    let tile_index_xyz = tile_corner.x +
        tile_corner.y * size64.x +
        tile_corner.z * size64.x * size64.y;

    // Reset this tile's position in the tape tree
    tile_tape[tile_index_xyz] = 0u;

    // Tile's lower z position, in voxels
    let corner_z = tile_corner.z * TILE_SIZE;

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

    var new_tape_start = 0u;
    if stack.has_choice {
        new_tape_start = simplify_tape(out.pos, out.count, &stack);
    }

    tile_tape[tile_index_xyz] = new_tape_start;

    // The tile is full, so set the "filled" flag when pushing the tile to the
    // tape list, which short-circuits evaluation.  We do this instead of just
    // setting tile_zmin so that the tile is evaluated when rendering normals,
    // because we need to compute normals for filled pixels in root tiles.
    var filled_bit = 0u;
    if v[1] < 0.0 {
        filled_bit = 1 << 31u;
    }

    // Select the active strata, based on Z position
    let strata_size = strata_size_bytes() / 4; // bytes -> words
    let i = strata_size * tile_corner.z;

    // `count` is at offset 3 in the struct
    let offset = atomicAdd(&tiles_out[i + 3], 1u);

    // the actual tile index is somewhere past the 4th word
    atomicStore(&tiles_out[i + 4 + offset], tile_index_xyz | filled_bit);

    // write the workgroup sizes to the first 3 words in the `struct`
    let count = offset + 1u;
    let wg_dispatch_x = min(count, 32768u);
    let wg_dispatch_y = (count + 32767u) / 32768u;
    atomicMax(&tiles_out[i], wg_dispatch_x);
    atomicMax(&tiles_out[i + 1], wg_dispatch_y);
    atomicMax(&tiles_out[i + 2], 1u);
}

fn next_multiple_of(a: u32, b: u32) -> u32 {
    return ((a + (b - 1)) / b) * b;
}

/// Per-strata offset in the root tiles list
///
/// This must be equivalent to `strata_size_bytes` in the Rust code
fn strata_size_bytes() -> u32 {
    let nx = config.render_size.x / 64u;
    let ny = config.render_size.y / 64u;

    // Each strata has a [vec3u, u32] header, adding 4 words
    let size_words = nx * ny + 4u;
    let size_bytes = size_words * 4u;
    return next_multiple_of(size_bytes, 256);
}

/// Allocates a new chunk, returning a past-the-end pointer
///
/// Note that we increment the root tape length, because we want to preserve
/// these tapes through the reset in `backfill`
fn alloc(chunk_size: u32) -> u32 {
    atomicAdd(&config.tape_data_offset, chunk_size);
    return atomicAdd(&config.root_tape_len, chunk_size);
}

fn dealloc(chunk_size: u32) {
    atomicSub(&config.tape_data_offset, chunk_size);
    atomicSub(&config.root_tape_len, chunk_size);
}
