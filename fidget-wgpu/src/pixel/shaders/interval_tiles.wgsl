// Interval evaluation stage for pixel rasterization

/// Per-state IO bindings
@group(2) @binding(0) var<storage, read> tiles_in: TileListInput;

@group(2) @binding(1) var<storage, read_write> subtiles_out: TileListOutput;
@group(2) @binding(2) var<storage, read_write> subtile_values: array<RawDistancePixel>;

/// Input tile size; one input tile maps to a 8x8 workgroup
const TILE_SIZE: u32 = 64;
const SUBTILE_SIZE: u32 = 8;

@compute @workgroup_size(8, 8)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // We dispatch with workgroups only on the X axis
    for (var i=workgroup_id.x; i < tiles_in.count; i += num_workgroups.x) {
        interval_tile_worker(i, local_id.xy);
    }
}

fn interval_tile_worker(
    active_tile_index: u32,
    local_id: vec2u
) {
    // Convert to a size in tile units
    let size_tiles = config.render_size / TILE_SIZE;
    let size_subtiles = size_tiles * 8u;

    // Get global tile position, in tile coordinates.  The top bit indicates
    // that the tile is filled.
    let t = tiles_in.active_tiles[active_tile_index];
    let tx = t % size_tiles.x;
    let ty = (t / size_tiles.x) % size_tiles.y;
    let tile_corner = vec2u(tx, ty);

    // Subtile corner position
    let subtile_corner = tile_corner * 4 + local_id;
    let subtile_index_xy = subtile_corner.x + subtile_corner.y * size_subtiles.x;

    // Subtile corner position, in voxels
    let corner_pos = subtile_corner * SUBTILE_SIZE;

    // Compute transformed interval regions (XY only)
    let m_xy = interval_inputs(tile_corner, TILE_SIZE);

    // Patch in the Z value from the render config
    let m = array(m_xy[0], m_xy[1], build_imm(config.z));

    // Do the actual interpreter work
    var stack = Stack();
    let tape_offset = get_tape_offset_for_level(corner_pos, TILE_SIZE);
    var tape_start = tile_tape[tape_offset];
    let out = run_tape(tape_start, m, &stack);

    let v = out.value.v;
    if v[1] < 0.0 {
        // write a full distance pixel
        subtile_values[subtile_index_xy] = distance_pixel_fill(1, true);
        return;
    } else if v[0] > 0.0 {
        // write an empty distance pixel
        subtile_values[subtile_index_xy] = distance_pixel_fill(1, false);
        return;
    }

    // Push this subtile to the output list
    let offset = atomicAdd(&subtiles_out.count, 1u);
    let subtile_index_xyz = subtile_corner.x +
        (subtile_corner.y * size_subtiles.x);
    subtiles_out.active_tiles[offset] = subtile_index_xy;

    // TODO figure dispatch size for pixel evaluation?

    let next = simplify_tape(out.pos, out.count, &stack);
    if next != 0 {
        tape_start = next;
    }
    let next_tape_offset = get_tape_offset_for_level(corner_pos, SUBTILE_SIZE);
    tile_tape[next_tape_offset] = tape_start;
}

/// Allocates a new chunk, returning a past-the-end pointer
fn alloc(chunk_size: u32) -> u32 {
    return atomicAdd(&config.tape_data_offset, chunk_size);
}

fn dealloc(chunk_size: u32) {
    atomicSub(&config.tape_data_offset, chunk_size);
}
