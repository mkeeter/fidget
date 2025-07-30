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

/// Tile data
///
/// This is a dense grid of 64x64x64 tile data, which is either
/// - `u32::MAX` (empty)
/// - A value with bit 32 set (full, with tape index in bits 0-31)
/// - A value with bit 32 cleared (ambiguous, with tape index in bits 0-31)
@group(0) @binding(2) var<storage, read> tile64_tapes: array<u32>;

/// Active tiles, as global tile positions
@group(0) @binding(3) var<storage, read> active_tile64: array<u32>;

@group(0) @binding(4) var<storage, read_write> tile64_next: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile64_occupancy: array<array<u32, 4>>;

/// tile16 outputs
@group(0) @binding(6) var<storage, read_write> active_tile16_count: array<atomic<u32>, 3>;
@group(0) @binding(7) var<storage, read_write> active_tile16: array<u32>;
@group(0) @binding(8) var<storage, read_write> tile16_occupancy: array<array<u32, 4>>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Number of active tiles in `active_tile64`
    active_tile_count: u32,

    /// Image size, in voxel units
    image_size: vec3u,

    /// Number of slots in the output buffer
    out_buffer_size: u32,
}

var<workgroup> wg_occupancy: array<atomic<u32>, 4>;
var<workgroup> wg_offset: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile16_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile64_index = workgroup_id.x + workgroup_id.y * 65536;

    if (active_tile64_index >= config.active_tile_count) {
        return;
    }

    // Convert to a size in tile units
    let size64 = (config.image_size + 63u) / 64;
    let size16 = size64 * 4u;

    // Get global tile position, in tile64 coordinates (which corresponds to
    // position in the tile64_* arrays)
    let t = active_tile64[active_tile64_index];
    let tx = t % size64.x;
    let ty = (t / size64.x) % size64.y;
    let tz = (t / (size64.x * size64.y)) % size64.z;
    let tile64_corner = vec3u(tx, ty, tz);

    // Subtile offset within the tile
    let tile16_offset = vec3u(local_id.xyz);

    // Pick out our starting tape. `active_tile64` only contains truly active
    // tiles, so we don't need to check for the empty / full case.
    let tape_start = tile64_tapes[t];

    // Subtile corner position, in voxels
    let corner_pos = tile64_corner * 64 + tile16_offset * 16;

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + 16));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + 16));
    let iz = vec2f(f32(corner_pos.z), f32(corner_pos.z + 16));
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
    let out = run_tape_i(tape_start, m);

    // Figure out which 2 bits to touch in the occupancy array
    let bit_index = 2 * (tile16_offset.z + tile16_offset.y * 4 + tile16_offset.x * 16);
    if (out[1] < 0.0) {
        // Full = 0b01
        atomicOr(&wg_occupancy[bit_index / 32], 1u << (bit_index % 32));
    } else if (out[0] > 0.0) {
        // Empty = 0b10
        atomicOr(&wg_occupancy[bit_index / 32], 2u << (bit_index % 32));
    } else {
        // Ambiguous = 0b11
        atomicOr(&wg_occupancy[bit_index / 32], 3u << (bit_index % 32));
    }

    workgroupBarrier();
    var occupancy = array<u32, 4>(0u, 0u, 0u, 0u);
    for (var i=0; i < 4; i++) {
        occupancy[i] = atomicLoad(&wg_occupancy[i]);
    }
    if (local_id.y == 0 && local_id.z == 0) {
        tile64_occupancy[t][local_id.x] = occupancy[local_id.x];
    }
    let o = Occupancy(occupancy);
    let total_offset = occupancy_size(o);

    // Allocate memory in the 0,0,0 subtile thread
    if (bit_index == 0u) {
        var prev_x = atomicAdd(&active_tile16_count[0], total_offset);
        var prev_y = 0u;
        if (prev_x + total_offset >= 65536u) {
            atomicSub(&active_tile16_count[0], 65536u);
            prev_y = atomicAdd(&active_tile16_count[1], 1u);
        } else {
            prev_y = atomicLoad(&active_tile16_count[1]);
        }
        wg_offset = prev_x + prev_y * 65536u;
    }
    workgroupBarrier();

    let offset = wg_offset;
    if (offset + total_offset < config.out_buffer_size) {
        let local_offset = offset + occupancy_offset(o, bit_index / 2u);
        let tile16_corner = tile64_corner * 4 + tile16_offset;
        let subtile_index = tile16_corner.x +
            (tile16_corner.y * size16.x) +
            (tile16_corner.z * size16.x * size16.y);
        active_tile16[local_offset] = subtile_index;
        for (var i=0; i < 4; i++) {
            tile16_occupancy[local_offset][i] = 0u;
        }
        if (bit_index == 0u) {
            tile64_next[t] = offset;
        }
    } else {
        if (bit_index == 0u) {
            tile64_next[t] = 0xFFFFFFFFu;
        }
    }
}
