@group(1) @binding(0) var<storage, read> tiles_out: TileListInput;
@group(1) @binding(1) var<storage, read> tile64_zmax: array<u32>;

// This is a set of per-strata `TileListOutput` arrays.  Each one is
// `strata_size_bytes(..)` long, which is large enough to fit every tile.  We
// can't represent this directly, so good luck poking the right memory locations
// by hand!
@group(1) @binding(2) var<storage, read_write> strata_tiles: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn repack_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let index = global_id.x +
        global_id.y * num_workgroups.x * 64u +
        global_id.z * num_workgroups.x * num_workgroups.y * 64u;
    if index >= tiles_out.count {
        return;
    }

    // Calculate render size in tile units
    let size64 = config.render_size / 64;

    // Read the tile and clear the filled bit
    let t = tiles_out.active_tiles[index] & 0x7FFFFFFF;

    // Unpack into XY and Z components
    let tz = t / (size64.x * size64.y);
    let tile_index_xy = t % (size64.x * size64.y);

    // Figure out how far this tile is from the zmax tile in the XY position
    let zmax = tile64_zmax[tile_index_xy];
    let strata = zmax - tz;

    // Select the active strata, based on strata depth from zmax
    let strata_size = strata_size_bytes() / 4; // bytes -> words
    let i = strata_size * strata;

    // `count` is at offset 3 in the struct
    let offset = atomicAdd(&strata_tiles[i + 3], 1u);

    // the actual tile index is somewhere past the 4th word
    strata_tiles[i + 4 + offset] = t;

    // Write the workgroup sizes to the first 3 words in the `struct`
    // We dispatch a maximum of [32768, 1, 1] and iterate in the shader
    let count = offset + 1u;
    let wg_dispatch_x = min(count, 32768u);
    atomicMax(&strata_tiles[i], wg_dispatch_x);
    atomicMax(&strata_tiles[i + 1], 1u);
    atomicMax(&strata_tiles[i + 2], 1u);
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

    // Snap to `min_storage_buffer_offset_alignment`
    return next_multiple_of(size_bytes, 256);
}

