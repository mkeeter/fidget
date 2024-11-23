// VM interpreter for floating-point values, using tiled rendering
//
// This shader must also be concatenated with `interpreter_4f.wgsl`, which
// provides the `run_tape` function.

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
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Tile size, in pixels
    tile_size: u32,

    /// Window size, in pixels
    window_size: vec2u,

    /// Number of tiles to render
    tile_count: u32,

    /// Explicit padding
    _padding: u32,
}

/// Per-tile render configuration
struct Tile {
    /// Corner of the tile, as a pixel position
    corner: vec2u,

    /// Starting point of this tile in the `tape` array
    start: u32,

    /// Explicit padding
    _padding: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let tile_idx = id.x / (config.tile_size / 4 * config.tile_size);

    // Position within the tile
    let pos_x = 4 * (id.x % (config.tile_size / 4)); // 4x SIMD
    let pos_y = (id.x / (config.tile_size / 4)) % config.tile_size;

    if (tile_idx < config.tile_count) {
        let tile = tiles[tile_idx];
        if (pos_x < config.tile_size && pos_y < config.tile_size) {
            // Dummy value for inputs
            var m = mat4x4f(
                vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)
            );

            for (var i=0u; i < 4; i += 1u) {
                // Absolute pixel position
                let corner_pixels = vec2f(tile.corner + vec2(pos_x + i, pos_y));

                let pos_pixels = vec4f(corner_pixels, 0.0, 0.0);
                let pos_model = config.mat * pos_pixels;

                var v = vec4f(0.0);
                v[config.axes.x] = pos_model.x;
                v[config.axes.y] = pos_model.y;
                m[i] = v;
            }

            let out = run_tape(tile.start, m);
            for (var i = 0u; i < 4; i += 1u) {
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
