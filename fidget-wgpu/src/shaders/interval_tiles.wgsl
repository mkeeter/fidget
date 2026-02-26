// Interval evaluation stage for raymarching shader
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

@group(1) @binding(0) var<storage, read_write> dispatch_counter: array<u32>;
@group(1) @binding(1) var<storage, read> tiles_in: TileListInput;
@group(1) @binding(2) var<storage, read> tile_zmin: array<Voxel>;

@group(1) @binding(3) var<storage, read_write> tape_data: TapeData;

@group(1) @binding(4) var<storage, read_write> subtiles_out: TileListOutput;
@group(1) @binding(5) var<storage, read_write> subtile_zmin: array<Voxel>;
@group(1) @binding(6) var<storage, read_write> subtile_hist: array<atomic<u32>>;

/// Input tile size; one input tile maps to a 4x4x4 workgroup
override TILE_SIZE: u32;

/// Output tile size, must be TILE_SIZE / 4; one output tile maps to one thread
override SUBTILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    // Handle the dispatch counter update.  In theory, we just need a single
    // counter; in practice, we need one per workgroup because we can't do
    // global synchronization of the update.
    //
    // We know that each dispatch is of size `max_tiles_per_dispatch` except the
    // final one, so this gives us a tile index to process (which is the global
    // dispatch offset plus a workgroups-specific offset).
    var workgroup_index = workgroup_id.x; // always 1D
    let tile_in_index = workgroup_index +
        dispatch_counter[workgroup_index] * config.max_tiles_per_dispatch;
    workgroupBarrier();

    // Updating the counter is only done by thread 0 in the workgroup.
    if local_id.x == 0 && local_id.y == 0 && local_id.z == 0 {
        // If this dispatch finishes processing all tiles, then reset the
        // counter to prepare for the next pass.
        if tile_in_index + num_workgroups.x >= tiles_in.count {
            dispatch_counter[workgroup_index] = 0u;
        } else {
            dispatch_counter[workgroup_index] += 1u;
        }
    }

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Get global tile position, in tile coordinates.
    if tile_in_index >= tiles_in.count {
        return;
    }
    let tile = tiles_in.active_tiles[tile_in_index];
    let t = tile.tile;
    let tx = t % size_tiles.x;
    let ty = (t / size_tiles.x) % size_tiles.y;
    let tz = (t / (size_tiles.x * size_tiles.y)) % size_tiles.z;
    let tile_corner = vec3u(tx, ty, tz);

    // Subtile corner position
    let subtile_corner = tile_corner * 4 + local_id;
    let subtile_index_xy = subtile_corner.x + subtile_corner.y * size_subtiles.x;
    let subtile_index_xyz = subtile_index_xy + subtile_corner.z * size_subtiles.x * size_subtiles.y;

    // Subtile corner position, in voxels
    let corner_pos = subtile_corner * SUBTILE_SIZE;

    // Check for Z masking from parent tile
    let tile_index_xy = tile_corner.x + tile_corner.y * size_tiles.x;
    let tile_value = tile_zmin[tile_index_xy].value;
    if (tile_value >> 20) >= corner_pos.z + SUBTILE_SIZE - 1 {
        atomicMax(&subtile_zmin[subtile_index_xy].value, tile_value);
        return;
    }

    // Last-minute check to see if anyone filled out this tile
    if (atomicLoad(&subtile_zmin[subtile_index_xy].value) >> 20) >= corner_pos.z + SUBTILE_SIZE {
        return;
    }

    // Compute transformed interval regions
    let m = interval_inputs(subtile_corner, SUBTILE_SIZE);

    // Do the actual interpreter work
    var stack = Stack();
    var tape_start = tile.tape_index;
    let out = run_tape(tape_start, m, &stack);

    let v = out.value.v;

    // If the tile is completely empty, then we're done!
    if v[0] > 0.0 {
        return;
    }

    let next = simplify_tape(out.pos, out.count, &stack);
    if next != 0 {
        tape_start = next;
    }

    if v[1] < 0.0 {
        // Full, write to subtile_zmin but don't return yet (because we want to
        // store a simplified tape for normal evaluation)
        // CAS loop, see interval_root for details
        let new_z = corner_pos.z + SUBTILE_SIZE - 1;
        let new_value = (new_z << 20) | (tape_start / 2); // XXX handle overflow?
        atomicMax(&subtile_zmin[subtile_index_xy].value, new_value);
    } else {
        // Otherwise, enqueue the tile and add its Z position to the histogram
        let offset = atomicAdd(&subtiles_out.count, 1u);
        subtiles_out.tiles[offset] = ActiveTile(subtile_index_xyz, tape_start);
        atomicAdd(&subtile_hist[subtile_corner.z], 1);
    }
}
