// VM interpreter for floating-point values, using tiled rendering
//
// `OP_*` constants are generated at runtime based on bytecode format, so this
// shader cannot be compiled as-is.  This shader must also be concatenated with
// `interpreter_4f.wgsl`, which provides the `run_tape` function.

/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Tiles to render
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;

/// Input tape(s), serialized to bytecode
@group(0) @binding(2) var<storage, read> tape: array<u32>;

/// Output array (single values), of same length as `vars`
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-world transform matrix, converting pixels to ±1
    screen_to_world: mat4x4<f32>,

    /// World-to-model transform matrix, converting ±1 to model space
    world_to_model: mat4x4<f32>,

    /// Mapping from X, Y, Z to input indices
    axes: vec3<u32>,

    /// Tile size, in pixels
    tile_size: u32,

    /// Window size, in pixels
    window_size: vec2<u32>,

}

/// Per-tile render configuration
struct Tile {
    /// Corner of the tile, as a pixel position
    corner: vec2<u32>,

    /// Starting point of this tile in the `tape` array
    start: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tile_idx = id.x / (config.tile_size / 4 * config.tile_size);

    // Position within the tile
    let pos_x = 4 * (id.x % (config.tile_size / 4)); // 4x SIMD
    let pos_y = (id.x / (config.tile_size / 4)) % config.tile_size;

    if (tile_idx < arrayLength(&tiles)) {
        let tile = tiles[tile_idx];
        if (pos_x < config.tile_size && pos_y < config.tile_size) {
            // Dummy value for inputs
            var m = mat4x4<f32>(
                vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
            );

            for (var i=0u; i < 4; i += 1u) {
                // Absolute pixel position
                let corner_pixels = tile.corner + vec2(pos_x + i, pos_y);

                let pos_pixels = vec4f(f32(corner_pixels.x), f32(corner_pixels.y), 0.0, 0.0);
                let pos_world = config.screen_to_world * pos_pixels;
                let pos_model = config.world_to_model * pos_world;

                var v = vec4<f32>(0.0);
                v[config.axes.x] = pos_model.x;
                v[config.axes.y] = pos_model.y;
                m[i] = v;
            }

            let out = run_tape(tile.start, m);
            for (var i=0u; i < 4; i += 1u) {
                var p = 0u;
                if (out[i] < 0.0) {
                    p = 0xFFFFFFFFu;
                } else {
                    p = 0xFF000000u;
                };

                // Write to absolute position in the image
                let pos_pixels = tile.corner + vec2(pos_x + i, pos_y);
                result[pos_pixels.x + pos_pixels.y * config.window_size.x] = p;
            }
        }
    }
}
