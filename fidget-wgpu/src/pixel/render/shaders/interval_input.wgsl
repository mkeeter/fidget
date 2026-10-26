fn interval_inputs(tile_corner: vec2u, tile_size: u32) -> array<Value, 2> {
    // Tile corner position, in voxels
    let corner_pos = tile_corner * tile_size;

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + tile_size));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + tile_size));

    return transformed_inputs(Value(ix), Value(iy));
}
