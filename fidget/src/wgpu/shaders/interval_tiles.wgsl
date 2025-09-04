// Interval evaluation stage for raymarching shader
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

/// Per-state IO bindings
@group(1) @binding(0) var<storage, read_write> tiles_in: TileListInput;
@group(1) @binding(1) var<storage, read> tile_zmin: array<u32>;

@group(1) @binding(2) var<storage, read_write> subtiles_out: TileListOutput;
@group(1) @binding(3) var<storage, read_write> subtile_zmin: array<atomic<u32>>;

/// Input tile size; one input tile maps to a 4x4x4 workgroup
override TILE_SIZE: u32;

/// Output tile size, must be TILE_SIZE / 4; one output tile maps to one thread
override SUBTILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile_index = workgroup_id.x + workgroup_id.y * 32768;
    if active_tile_index >= tiles_in.count {
        return;
    }

    // Prepare for normal evaluation by tweaking the dispatch size.  This gets
    // applied to each tile size, but only matters when we're dispatching the
    // normal shader.
    tiles_in.wg_size[2] = 64u;

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Get global tile position, in tile coordinates.  The top bit indicates
    // that the tile is filled.
    let t_raw = tiles_in.active_tiles[active_tile_index];
    let t_filled = (t_raw & (1 << 31u)) != 0;
    let t = t_raw & 0x7FFFFFFF;
    let tx = t % size_tiles.x;
    let ty = (t / size_tiles.x) % size_tiles.y;
    let tz = (t / (size_tiles.x * size_tiles.y)) % size_tiles.z;
    let tile_corner = vec3u(tx, ty, tz);

    // Subtile corner position
    let subtile_corner = tile_corner * 4 + local_id;
    let subtile_index_xy = subtile_corner.x + subtile_corner.y * size_subtiles.x;

    // Subtile corner position, in voxels
    let corner_pos = subtile_corner * SUBTILE_SIZE;

    // Special handling for uniformly filled tiles
    if t_filled {
        // Snap down to the larger tile size
        let z = (corner_pos.z / TILE_SIZE) * TILE_SIZE;
        atomicMax(&subtile_zmin[subtile_index_xy], z + TILE_SIZE - 1);
        return;
    }

    // Check for Z masking from tile
    let tile_index_xy = tile_corner.x + tile_corner.y * size_tiles.x;
    if tile_zmin[tile_index_xy] >= corner_pos.z {
        atomicMax(&subtile_zmin[subtile_index_xy], tile_zmin[tile_index_xy]);
        return;
    }

    // Compute transformed interval regions
    let m = interval_inputs(subtile_corner, SUBTILE_SIZE);

    // Last-minute check to see if anyone filled out this tile
    if atomicLoad(&subtile_zmin[subtile_index_xy]) >= corner_pos.z + SUBTILE_SIZE {
        return;
    }

    // Do the actual interpreter work
    let tape_start = get_tape_start(corner_pos);
    var stack = Stack();
    let out = run_tape(tape_start, m, &stack);

    if !out.valid {
        return;
    }

    let v = out.value.v;
    if v[1] < 0.0 {
        // Full, write to subtile_zmin
        atomicMax(&subtile_zmin[subtile_index_xy], corner_pos.z + SUBTILE_SIZE - 1);
    } else if v[0] > 0.0 {
        // Empty, nothing to do here
    } else {
        let offset = atomicAdd(&subtiles_out.count, 1u);
        let subtile_index_xyz = subtile_corner.x +
            (subtile_corner.y * size_subtiles.x) +
            (subtile_corner.z * size_subtiles.x * size_subtiles.y);
        subtiles_out.active_tiles[offset] = subtile_index_xyz;

        let count = offset + 1u;
        let wg_dispatch_x = min(count, 32768u);
        let wg_dispatch_y = (count + 32767u) / 32768u;
        atomicMax(&subtiles_out.wg_size[0], wg_dispatch_x);
        atomicMax(&subtiles_out.wg_size[1], wg_dispatch_y);
        atomicMax(&subtiles_out.wg_size[2], 1u);
    }
}

/// Allocates a new chunk, returning a past-the-end pointer
fn alloc(chunk_size: u32) -> u32 {
    return atomicAdd(&config.tape_data_offset, chunk_size);
}

fn dealloc(chunk_size: u32) {
    atomicSub(&config.tape_data_offset, chunk_size);
}
