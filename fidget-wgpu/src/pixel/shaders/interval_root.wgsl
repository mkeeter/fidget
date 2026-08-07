// Interval root tile evaluation

@group(2) @binding(0) var<storage, read_write> tiles_out: TileListOutput;
@group(2) @binding(1) var<storage, read_write> tile_values: array<RawDistancePixel>;

/// Root tile size
const TILE_SIZE: u32 = 64;

@compute @workgroup_size(8, 8)
fn interval_root_main(
    @builtin(global_invocation_id) tile_corner_: vec3u
) {
    // We only care about XY position
    let tile_corner = tile_corner_.xy;

    // Calculate render size in tile units
    let size64 = config.render_size / 64;

    if (tile_corner.x >= size64.x ||
        tile_corner.y >= size64.y)
    {
        return;
    }

    let tile_index_xy = tile_corner.x + tile_corner.y * size64.x;

    // Tile's lower corner position, in voxels
    let corner_pos = tile_corner * TILE_SIZE;

    // Compute transformed interval regions (XY only)
    let m_xy = interval_inputs(tile_corner, TILE_SIZE);

    // Patch in the Z value from the render config
    let m = array(m_xy[0], m_xy[1], build_imm(config.z));

    // Do the actual interpreter work
    var stack = Stack();
    let out = run_tape(0u, m, &stack);
    let v = out.value.v;

    // If the tile is completely empty or full, then we write an appropriate
    // value to the output and return immediately.
    if v[0] > 0.0 && config.pixel_perfect != 0 {
        tile_values[tile_index_xy] = distance_pixel_fill(0, false);
        return;
    } else if v[1] < 0.0 && config.pixel_perfect != 0 {
        tile_values[tile_index_xy] = distance_pixel_fill(0, true);
        return;
    }

    // We have to subdivide and recurse, which we do by writing the 64^2
    // tile and incrementing our dispatch size
    let offset = atomicAdd(&tiles_out.count, 1u);
    tiles_out.active_tiles[offset] = tile_index_xy;

    let next = simplify_tape(out.pos, out.count, &stack);
    if next != 0 {
        // Update this tile's position in the tape index map
        let tape_index = get_tape_offset_for_level(corner_pos, 64u);
        tile_tape[tape_index] = next;
    }

    // We dispatch a maximum of [32768, 1, 1] and iterate in the shader
    let count = offset + 1u;
    let wg_dispatch_x = min(count, 32768u);
    atomicMax(&tiles_out.wg_size[0], wg_dispatch_x);
    atomicMax(&tiles_out.wg_size[1], 1u);
    atomicMax(&tiles_out.wg_size[2], 1u);
}

/// Allocates a new chunk, returning a past-the-end pointer
fn alloc(chunk_size: u32) -> u32 {
    return atomicAdd(&config.tape_data_offset, chunk_size);
}

/// Undo an allocation
fn dealloc(chunk_size: u32) {
    atomicSub(&config.tape_data_offset, chunk_size);
}
