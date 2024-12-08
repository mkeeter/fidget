// VM interpreter for floating-point values, using voxel tiles
//
// This shader must also be concatenated with `interpreter_4f.wgsl`, which
// provides the `run_tape` function, and `common.wgsl`, which provides common
// bindings (0-2) and types.

/// Tiles to render
@group(0) @binding(3) var<storage, read> tiles: array<Tile3D>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let tile_idx = id.x / (config.tile_size * config.tile_size / 4);

    // Position within the tile
    let pos_x = 4 * (id.x % (config.tile_size / 4)); // 4x SIMD
    let pos_y = (id.x / (config.tile_size / 4)) % config.tile_size;

    if (tile_idx < config.tile_count) {
        let tile = tiles[tile_idx];

        // Build up input map
        var m = mat4x4f(
            vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)
        );

        for (var i=0u; i < 4; i += 1u) {
            // Absolute pixel position
            let corner_pixels = vec2f(tile.corner + vec2(pos_x + i, pos_y));

            let pos_pixels = vec4f(corner_pixels, 0.0, 1.0);
            let pos_model = config.mat * pos_pixels;

            m[config.axes.x][i] = pos_model.x;
            m[config.axes.y][i] = pos_model.y;
        }

        // Do the actual interpreter work
        let out = run_tape(tile.start, m);

        // Unpack the 4x results into pixels
        for (var i = 0u; i < 4; i += 1u) {
            var p = 0u;
            if (out[i] < 0.0) {
                p = 0xFFFFFFFFu;
            } else {
                p = 0xFF000000u;
            }

            // Write to absolute position in the image
            let pos_pixels = tile.corner + vec2(pos_x + i, pos_y);
            result[pos_pixels.x + pos_pixels.y * config.window_size.x] = p;
        }
    }
}

