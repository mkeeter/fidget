// VM interpreter for floating-point values, using voxel tiles

@group(2) @binding(0) var<storage, read> tiles_in: TileListInput;
@group(2) @binding(1) var<storage, read> tile4_zmin: array<u32>;

/// Output array, render size (image size rounded up to multiple of 64 voxels)
@group(2) @binding(2) var<storage, read_write> result: array<atomic<u32>>;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // We dispatch with workgroups only on the X axis
    for (var i=workgroup_id.x; i < tiles_in.count; i += num_workgroups.x) {
        voxel_tile_worker(i, local_id);
    }
}

fn voxel_tile_worker(
    active_tile4_index: u32,
    local_id: vec3u
) {
    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;

    // Get global tile position, in tile4 coordinates
    let t = tiles_in.active_tiles[active_tile4_index];
    let tx = t % size4.x;
    let ty = (t / size4.x) % size4.y;
    let tz = (t / (size4.x * size4.y)) % size4.z;
    let tile4_corner = vec3u(tx, ty, tz);

    // Subtile corner position, in voxels
    let corner_pos = tile4_corner * 4 + local_id;

    let tile4_index_xy = tx + ty * size4.x;
    let pixel_index_xy = corner_pos.x + corner_pos.y * config.render_size.x;
    if tile4_zmin[tile4_index_xy] >= corner_pos.z {
        atomicMax(&result[pixel_index_xy], tile4_zmin[tile4_index_xy]);
        return;
    }

    // Last chance to bail out
    if atomicLoad(&result[pixel_index_xy]) >= u32(corner_pos.z) {
        return;
    }

    // Compute input values
    let m = transformed_inputs(
        Value(f32(corner_pos.x)),
        Value(f32(corner_pos.y)),
        Value(f32(corner_pos.z)),
    );

    // Do the actual interpreter work
    let tape_offset = get_tape_offset_for_level(corner_pos, 4);
    let tape_start = tile_tape[tape_offset];
    var stack = Stack(); // dummy value
    let out = run_tape(tape_start, m, &stack);

    if out.value.v < 0.0 {
        atomicMax(&result[pixel_index_xy], corner_pos.z);
    }
}
