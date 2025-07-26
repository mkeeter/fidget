// Interval evaluation stage for raymarching shader
//
// This shader must also be concatenated with `interpreter_i.wgsl`, which
// provides the `run_tape` function.

/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Tile data
///
/// This is a dense grid of 64x64x64 tile data, which is either
/// - `u32::MAX` (empty)
/// - A value with bit 32 set (full, with tape index in bits 0-31)
/// - A value with bit 32 cleared (ambiguous, with tape index in bits 0-31)
@group(0) @binding(1) var<storage, read> dense_tile64: array<u32>;

/// Input tape(s), serialized to bytecode
///
/// This array contains many tapes concatenated together; tape start positions
/// are recorded in the `dense_tile64` input array.
@group(0) @binding(2) var<storage, read> tape: array<u32>;

/// Active tiles, as indices into `dense_tile64`
@group(0) @binding(3) var<storage, read> active_tiles: array<u32>;

/// Output array, a densely-packed map of occupancy maps
@group(0) @binding(4) var<storage, read_write> dense_tile64_occupancy: array<array<atomic<u32>, 32>>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Number of tiles to evaluate (from `active_tiles`)
    tile_count: u32,

    /// Image size, in 64x64x64 tiles
    image_size_tiles: vec3u,

    /// Explicit padding
    _padding: u32,
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile_index = workgroup_id.x + workgroup_id.y * 65535;

    if (active_tile_index >= config.tile_count) {
        return;
    }

    // Get global tile position, in tile coordinates
    let t = active_tiles[active_tile_index];
    let tx = t % config.image_size_tiles.x;
    let ty = (t / config.image_size_tiles.x) % config.image_size_tiles.y;
    let tz = (t / (config.image_size_tiles.x * config.image_size_tiles.y)) % config.image_size_tiles.z;
    let tile_corner = vec3u(tx, ty, tz);

    // Subtile offset within the tile
    let subtile_offset = vec3u(local_id.xy, workgroup_id.z);

    // Figure out which 2 bits to touch in the occupancy array
    let subtile_bit_index = 2 * (subtile_offset.z + subtile_offset.y * 8 + subtile_offset.x * 64);

    // Pick out our starting tape.  active_tiles only contains truly active
    // tiles, so we don't need to check for the empty / full case.
    let tape_start = dense_tile64[t];

    // Subtile corner position, in voxels
    let subtile_corner = tile_corner * 64 + subtile_offset * 8;

    // Compute transformed interval regions
    let ix = vec2f(f32(subtile_corner.x), f32(subtile_corner.x + 8));
    let iy = vec2f(f32(subtile_corner.y), f32(subtile_corner.y + 8));
    let iz = vec2f(f32(subtile_corner.z), f32(subtile_corner.z + 8));
    var ts = mat4x2f(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
    for (var i = 0; i < 4; i++) {
        ts[i] = mul_i(vec2f(config.mat[0][i]), ix)
            + mul_i(vec2f(config.mat[1][i]), iy)
            + mul_i(vec2f(config.mat[2][i]), iz)
            + vec2f(config.mat[3][i]);
    }

    // Build up input map
    var m = mat4x2f(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
    if (config.axes.x < 4) {
        m[config.axes.x] = div_i(ts[0], ts[3]);
    }
    if (config.axes.y < 4) {
        m[config.axes.y] = div_i(ts[1], ts[3]);
    }
    if (config.axes.z < 4) {
        m[config.axes.z] = div_i(ts[2], ts[3]);
    }

    // Do the actual interpreter work
    let out = run_tape(tape_start, m);

    if (out[1] < 0.0) {
        // Full = 0b01
        atomicOr(
            &dense_tile64_occupancy[t][subtile_bit_index / 32],
            1u << (subtile_bit_index % 32)
        );
    } else if (out[0] > 0.0) {
        // Empty = 0b10
        atomicOr(
            &dense_tile64_occupancy[t][subtile_bit_index / 32],
            2u << (subtile_bit_index % 32)
        );
    } else {
        // Ambiguous = 0b11
        atomicOr(
            &dense_tile64_occupancy[t][subtile_bit_index / 32],
            3u << (subtile_bit_index % 32)
        );
    }
}
