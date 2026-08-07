// VM interpreter for floating-point values, using voxel tiles
@group(2) @binding(0) var<storage, read> tiles_in: TileListInput;

/// Output array, as an image size (*not* rounded up)
@group(2) @binding(1) var<storage, read_write> result: array<RawDistancePixel>;

@compute @workgroup_size(8, 8)
fn pixel_tiles_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // We dispatch with workgroups only on the X axis
    for (var i=workgroup_id.x; i < tiles_in.count; i += num_workgroups.x) {
        pixel_tile_worker(i, local_id.xy);
    }
}

fn pixel_tile_worker(
    active_tile8_index: u32,
    local_id: vec2u
) {
    // Convert to a size in subtile units
    let size8 = config.render_size / 8;

    // Get global tile position, in tile8 coordinates
    let t = tiles_in.active_tiles[active_tile8_index];
    let tx = t % size8.x;
    let ty = (t / size8.x) % size8.y;
    let tile8_corner = vec2u(tx, ty);

    // Subtile corner position, in voxels
    let corner_pos = tile8_corner * 8 + local_id;

    // Skip pixels outside of the (un-rounded) image bounds
    if corner_pos.x >= config.image_size.x ||
       corner_pos.y >= config.image_size.y
    {
        return;
    }

    let pixel_index_xy = corner_pos.x + corner_pos.y * config.image_size.x;

    // Compute input values
    let m_xy = transformed_inputs(
        Value(f32(corner_pos.x)),
        Value(f32(corner_pos.y)),
    );
    let m = array(m_xy[0], m_xy[1], build_imm(config.z));

    // Do the actual interpreter work
    let tape_offset = get_tape_offset_for_level(corner_pos, 8);
    let tape_start = tile_tape[tape_offset];
    var stack = Stack(); // dummy stack
    let out = run_tape(tape_start, m, &stack);

    result[pixel_index_xy] = distance_pixel_value(out.value.v);
}
