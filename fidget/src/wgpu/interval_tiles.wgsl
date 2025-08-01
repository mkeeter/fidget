// Interval evaluation stage for raymarching shader
//
// This shader must also be concatenated with `interpreter_i.wgsl`, which
// provides the `run_tape` function.

/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Input tape(s), serialized to bytecode
///
/// This array contains many tapes concatenated together; tape start positions
/// are recorded in the `tile64_tapes` input array.
@group(0) @binding(1) var<storage, read> tape_data: array<u32>;
@group(0) @binding(2) var<storage, read> tile64_tapes: array<u32>;

@group(0) @binding(3) var<storage, read> active_tile_count: u32;
@group(0) @binding(4) var<storage, read> active_tile: array<u32>;
@group(0) @binding(5) var<storage, read> tile_zmin: array<u32>;

@group(0) @binding(6) var<storage, read_write> active_subtile_count: array<atomic<u32>, 4>;
@group(0) @binding(7) var<storage, read_write> active_subtile: array<u32>;
@group(0) @binding(8) var<storage, read_write> subtile_zmin: array<atomic<u32>>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,
    _padding1: u32,

    /// Image size, in voxels
    image_size: vec3u,
    _padding2: u32,
}

override TILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile_index = workgroup_id.x + workgroup_id.y * 32768;
    if (active_tile_index >= active_tile_count) {
        return;
    }

    // Convert to a size in tile units
    let size64 = (config.image_size + 63u) / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Get global tile position, in tile coordinates
    let t = active_tile[active_tile_index];
    let tx = t % size_tiles.x;
    let ty = (t / size_tiles.x) % size_tiles.y;
    let tz = (t / (size_tiles.x * size_tiles.y)) % size_tiles.z;
    let tile_corner = vec3u(tx, ty, tz);

    // Subtile corner position
    let subtile_corner = tile_corner * 4 + local_id;
    let subtile_index_xy = subtile_corner.x + subtile_corner.y * size_subtiles.x;

    // Subtile corner position, in voxels
    let corner_pos = subtile_corner * TILE_SIZE / 4;

    // Check for Z masking from tile
    let tile_index_xy = tile_corner.x + tile_corner.y * size_tiles.x;
    if (tile_zmin[tile_index_xy] >= corner_pos.z) {
        atomicMax(&subtile_zmin[subtile_index_xy], tile_zmin[tile_index_xy]);
        return;
    }

    // Pick out our starting tape, based on absolute position
    let tile64_pos = corner_pos / 64;
    let tile64_index = tile64_pos.x +
        tile64_pos.y * size64.x +
        tile64_pos.z * size64.x * size64.y;
    let tape_start = tile64_tapes[tile64_index];

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + TILE_SIZE / 4));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + TILE_SIZE / 4));
    let iz = vec2f(f32(corner_pos.z), f32(corner_pos.z + TILE_SIZE / 4));
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

    // Last-minute check to see if anyone filled out this tile
    if (atomicLoad(&subtile_zmin[subtile_index_xy]) >= corner_pos.z + TILE_SIZE / 4) {
        return;
    }

    // Do the actual interpreter work
    let out = run_tape_i(tape_start, m);

    if (out[1] < 0.0) {
        // Full, write to subtile_zmin
        atomicMax(&subtile_zmin[subtile_index_xy], corner_pos.z + TILE_SIZE / 4);
    } else if (out[0] > 0.0) {
        // Empty, nothing to do here
    } else {
        let offset = atomicAdd(&active_subtile_count[0], 1u);
        let subtile_index_xyz = subtile_corner.x +
            (subtile_corner.y * size_subtiles.x) +
            (subtile_corner.z * size_subtiles.x * size_subtiles.y);
        active_subtile[offset] = subtile_index_xyz;

        let count = offset + 1u;
        let wg_dispatch_x = min(count, 32768u);
        let wg_dispatch_y = (count + 32767u) / 32768u;
        atomicMax(&active_subtile_count[1], wg_dispatch_x);
        atomicMax(&active_subtile_count[2], wg_dispatch_y);
        atomicMax(&active_subtile_count[3], 1u);
    }
}
