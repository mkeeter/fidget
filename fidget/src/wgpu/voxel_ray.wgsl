// VM interpreter for floating-point values, using voxel tiles
//
// This shader must also be concatenated with `interpreter_4f.wgsl`, which
// provides the `run_tape` function.

/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Tile data (64^3, dense)
@group(0) @binding(1) var<storage, read> dense_tiles64: array<u32>;

/// Output array, a densely-packed map of occupancy maps
@group(0) @binding(2) var<storage, read> dense_tile64_occupancy: array<array<u32, 32>>;

/// Input tape(s), serialized to bytecode
@group(0) @binding(3) var<storage, read> tape: array<u32>;

/// Output array, image size
@group(0) @binding(4) var<storage, read_write> result: array<atomic<u32>>;

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

    /// Window size, in pixels
    image_size: vec3u,

    /// Explicit padding
    _padding2: u32,
}

const TILE_SIZE: u32 = 64;
const SUBTILE_SIZE: u32 = 8;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile and subtile index
    let tile = id.xy / TILE_SIZE;
    let subtile_offset = (id.xy / SUBTILE_SIZE) % 8;

    // Compute the number of tiles on each axis
    let nx = (config.image_size.x + TILE_SIZE - 1)/ TILE_SIZE;
    let ny = (config.image_size.y + TILE_SIZE - 1) / TILE_SIZE;

    // We start at the camera's position
    var z = i32(config.image_size.z) - 1 - i32(local_id.z) * 4;

    let pixel_index = id.x + id.y * config.image_size.x;
    var done = false;
    while (!done && z >= 0 && atomicLoad(&result[pixel_index]) == 0) {
        // Get the current tile that we're hanging out in
        let tz = u32(z) / TILE_SIZE;
        let i = tile.x + tile.y * nx + tz * nx * ny;
        let tape_start: u32 = dense_tiles64[i];
        if (tape_start == 0xFFFFFFFFu) {
            // Empty tile, keep going by jumping to the next tile boundary
            z -= 64;
            continue;
        } else if ((tape_start & 0x80000000) != 0) {
            // Full tile, we can break
            atomicMax(&result[pixel_index], u32(z));
            break;
        }

        // Iterate over subtiles in this tile
        for (var j=0; j < 4; j += 1) {
            let z_ = z - j * 16;
            let subtile_bit_index = 2 * ((u32(z_) / 8) % 8 + subtile_offset.y * 8 + subtile_offset.x * 64);
            let st = (dense_tile64_occupancy[i][subtile_bit_index / 32] >> (subtile_bit_index % 32)) & 3;
            if (st == 2) {
                // empty tile, keep going
            } else if (st == 1) {
                // Full tile, we can break
                atomicMax(&result[pixel_index], u32(z_));
                done = true;
                break;
            } else {
                // Voxel evaluation (or unpopulated, which shouldn't happen?)
                let out = raycast(tape_start, vec3u(id.xy, u32(z_)));
                atomicMax(&result[pixel_index], out);
                if (out > 0) {
                    done = true;
                    break;
                }
            }
        }
        z -= 64;
    }
}

fn raycast(tape_start: u32, pos: vec3u) -> u32 {
    // Build up input map
    var m = mat4x4f(
        vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)
    );

    for (var i=0u; i < 4; i += 1u) {
        let pos_pixels = vec4f(f32(pos.x), f32(pos.y), f32(pos.z - i), 1.0);
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
    let out = run_tape(tape_start, m);

    // Unpack the 4x results into pixels
    for (var i = 0u; i < 4; i += 1u) {
        if (out[i] < 0.0) {
            return pos.z - i;
        }
    }
    return 0u;
}
