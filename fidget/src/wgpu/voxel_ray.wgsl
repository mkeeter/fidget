// VM interpreter for floating-point values, using voxel tiles
//
// This shader must also be concatenated with `interpreter_4f.wgsl`, which
// provides the `run_tape` function.

/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Input tape(s), serialized to bytecode
@group(0) @binding(1) var<storage, read> tape_data: array<u32>;

/// Tile data (64^3, dense)
@group(0) @binding(2) var<storage, read> tile64_tapes: array<u32>;

@group(0) @binding(3) var<storage, read> active_tile4_count: array<u32, 4>;
@group(0) @binding(4) var<storage, read> active_tile4: array<u32>;

/// Output array, image size
@group(0) @binding(5) var<storage, read_write> result: array<atomic<u32>>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Explicit padding
    _padding1: u32,

    /// Window size, in voxels
    image_size: vec3u,

    /// Explicit padding
    _padding2: u32,
}

var<workgroup> wg_done: atomic<u32>;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile4_index = workgroup_id.x + workgroup_id.y * 32768;
    if (active_tile4_index >= active_tile4_count[0]) {
        return;
    }

    // Convert to a size in tile units
    let size64 = (config.image_size + 63u) / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;

    // Get global tile position, in tile4 coordinates
    let t = active_tile4[active_tile4_index];
    let tx = t % size4.x;
    let ty = (t / size4.x) % size4.y;
    let tz = (t / (size4.x * size4.y)) % size4.z;
    let tile4_corner = vec3u(tx, ty, tz);

    // Subtile corner position, in voxels
    let corner_pos = tile4_corner * 4 + local_id;

    // Pick out our starting tape, based on absolute position
    let tile64_pos = corner_pos / 64;
    let tile64_index = tile64_pos.x +
        tile64_pos.y * size64.x +
        tile64_pos.z * size64.x * size64.y;
    let tape_start = tile64_tapes[tile64_index];

    // Voxel evaluation
    let pixel_index = corner_pos.x + corner_pos.y * config.image_size.x;
    if (atomicLoad(&result[pixel_index]) >= u32(corner_pos.z)) {
        return;
    }
    let out = raycast(tape_start, vec3u(corner_pos));
    if (out > 0) {
        atomicMax(&result[pixel_index], out);
    }
}

fn raycast(tape_start: u32, pos: vec3u) -> u32 {
    // Build up input map
    var m = vec4f(0.0);

    let pos_pixels = vec4f(f32(pos.x), f32(pos.y), f32(pos.z), 1.0);
    let pos_model = config.mat * pos_pixels;

    if (config.axes.x < 4) {
        m[config.axes.x] = pos_model.x / pos_model.w;
    }
    if (config.axes.y < 4) {
        m[config.axes.y] = pos_model.y / pos_model.w;
    }
    if (config.axes.z < 4) {
        m[config.axes.z] = pos_model.z / pos_model.w;
    }

    // Do the actual interpreter work
    let out = run_tape(tape_start, m);
    if (out < 0.0) {
        return pos.z;
    } else {
        return 0u;
    }
}
