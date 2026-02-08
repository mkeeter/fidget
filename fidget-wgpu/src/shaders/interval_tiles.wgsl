// Interval evaluation stage for raymarching shader
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

@group(1) @binding(0) var<storage, read_write> dispatch_count: array<u32>;
@group(1) @binding(1) var<storage, read> dispatch: array<Dispatch>;
@group(1) @binding(2) var<storage, read> tiles_in: array<ActiveTile>;
@group(1) @binding(3) var<storage, read> tile_zmin: array<Voxel>;

@group(1) @binding(4) var<storage, read_write> tape_data: TapeData;

@group(1) @binding(5) var<storage, read_write> subtiles_out: TileListOutput;
@group(1) @binding(6) var<storage, read_write> subtile_zmin: array<Voxel>;
@group(1) @binding(7) var<storage, read_write> subtile_hist: array<atomic<u32>>;

/// Input tile size; one input tile maps to a 4x4x4 workgroup
override TILE_SIZE: u32;

/// Output tile size, must be TILE_SIZE / 4; one output tile maps to one thread
override SUBTILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Handle the dispatch counter update: we have a per-workgroup counter
    // (which should be eventually consistent across threads) because we can't
    // do global synchronization.  Thread 0 increments it by one after all
    // threads in the workgroup have had a chance to read it; if we're at the
    // end of dispatches, then we instead reset it to 0.
    let workgroup_index = workgroup_id.x
        + workgroup_id.y * num_workgroups.x
        + workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let d = dispatch_count[workgroup_index];
    workgroupBarrier();
    if local_index == 0u {
        if d + 1 == arrayLength(&dispatch) || dispatch[d + 1].tile_count == 0 {
            dispatch_count[workgroup_index] = 0;
        } else {
            dispatch_count[workgroup_index] += 1;
        }
    }
    workgroupBarrier();

    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.  This means that it's possible to
    // have more dispatches than active tiles.
    if workgroup_index >= dispatch[d].tile_count {
        return;
    }

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Get global tile position, in tile coordinates.  The top bit indicates
    // that the tile is filled.
    let tile = tiles_in[workgroup_index + dispatch[d].buffer_offset];
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
    if tile_zmin[tile_index_xy].z >= corner_pos.z + SUBTILE_SIZE {
        // CAS loop, see interval_root for details
        let new_z = tile_zmin[tile_index_xy].z;
        loop {
            let old_z = atomicLoad(&subtile_zmin[subtile_index_xy].z);
            if (new_z <= old_z) {
                break;
            }
            let exchanged_z = atomicCompareExchangeWeak(
                &subtile_zmin[subtile_index_xy].z, old_z, new_z);
            if (exchanged_z.exchanged) {
                // Z updated, now update the tape
                subtile_zmin[subtile_index_xy].tape_index = tile_zmin[tile_index_xy].tape_index;
                break;
            }
        }
        return;
    }

    // Last-minute check to see if anyone filled out this tile
    if atomicLoad(&subtile_zmin[subtile_index_xy].z) >= corner_pos.z + SUBTILE_SIZE {
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
        loop {
            let old_z = atomicLoad(&subtile_zmin[subtile_index_xy].z);
            if (new_z <= old_z) {
                break;
            }
            let exchanged_z = atomicCompareExchangeWeak(
                &subtile_zmin[subtile_index_xy].z, old_z, new_z);
            if (exchanged_z.exchanged) {
                // Z updated, now update the tape
                subtile_zmin[subtile_index_xy].tape_index = next;
                break;
            }
        }
    } else {
        // Otherwise, enqueue the tile and add its Z position to the histogram
        let offset = atomicAdd(&subtiles_out.count, 1u);
        subtiles_out.tiles[offset] = ActiveTile(subtile_index_xyz, next);
        atomicAdd(&subtile_hist[subtile_corner.z], 1);
    }
}
