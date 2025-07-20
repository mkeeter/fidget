// VM interpreter for floating-point values, using voxel tiles
//
// This shader must also be concatenated with `interpreter_4f.wgsl`, which
// provides the `run_tape` function, and `common.wgsl`, which provides common
// bindings and types.

// Each shader invocation evalutes 4 voxels, which only differ in X position
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let tile_idx = id.y + id.z * 65536;

    // Position within the tile
    let pos_x = 4 * (id.x % (config.tile_size / 4)); // 4x SIMD
    let pos_y = (id.x / (config.tile_size / 4)) % config.tile_size;
    let pos_z = (id.x / (config.tile_size * config.tile_size / 4)) % config.tile_size;

    if (tile_idx < config.tile_count) {
        let tile = tiles[tile_idx];

        // Build up input map
        var m = mat4x4f(
            vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)
        );

        for (var i=0u; i < 4; i += 1u) {
            // Absolute pixel position
            let corner_pixels = vec3f(tile.corner + vec3(pos_x + i, pos_y, pos_z));

            let pos_pixels = vec4f(corner_pixels, 1.0);
            let pos_model = config.mat * pos_pixels;

            if (config.axes.x < 4) {
                m[config.axes.x][i] = pos_model.x / pos_model.w;
            }
            if (config.axes.y < 4) {
                m[config.axes.y][i] = pos_model.y / pos_model.w;
            }
            if (config.axes.z < 4) {
                m[config.axes.z][i] = pos_model.z / pos_model.w;
            }
        }

        // Do the actual interpreter work
        let out = run_tape(tile.start, m);

        // Unpack the 4x results into pixels
        for (var i = 0u; i < 4; i += 1u) {
            if (out[i] < 0.0) {
                // Accumulate max height to absolute position in the tile
                let pos_pixels = tile.corner + vec3(pos_x + i, pos_y, pos_z);
                let j = pos_pixels.x + pos_pixels.y * config.window_size.x;
                atomicMax(&result[j], pos_pixels.z);
            }
        }
    }
}
