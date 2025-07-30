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
@group(0) @binding(3) var<storage, read> tile64_occupancy: array<array<u32, 4>>;
@group(0) @binding(4) var<storage, read> tile64_next: array<u32>;
@group(0) @binding(5) var<storage, read> tile16_occupancy: array<array<u32, 4>>;

/// Output array, image size
@group(0) @binding(6) var<storage, read_write> result: array<atomic<u32>>;

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

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile and subtile index
    let tile16 = global_id.xy / 16;
    let tile64 = global_id.xy / 64;
    let tile16_offset = (global_id.xy % 64) / 16;
    let tile4_offset = (global_id.xy % 16) / 4;

    // Compute the number of tiles on each axis
    let size64 = (config.image_size + 63u) / 64u;
    let size16 = (config.image_size + 15) / 16u;

    // We start at the camera's position
    var pos = vec3i(vec2i(global_id.xy), i32(config.image_size.z) - 1 - i32(local_id.z));

    let pixel_index = global_id.x + global_id.y * config.image_size.x;
    var done = false;
    while (!done && pos.z >= 0 && atomicLoad(&result[pixel_index]) == 0) {
        // Get the current tile that we're hanging out in
        let z64 = u32(pos.z) / 64;
        let tile64_index = tile64.x + tile64.y * size64.x + z64 * size64.x * size64.y;
        let tape_start: u32 = tile64_tapes[tile64_index];
        if (tape_start == 0xFFFFFFFFu) {
            // Empty tile, keep going by jumping to the next tile boundary
            pos.z -= 64;
            continue;
        } else if ((tape_start & 0x80000000) != 0) {
            // Full tile, we can break
            atomicMax(&result[pixel_index], u32(pos.z));
            break;
        }

        // Iterate over tile16 in this tile
        for (var j=0; j < 4 && !done; j += 1) {
            let bit_index = 2 * ((u32(pos.z) % 64) / 16 + tile16_offset.y * 4 + tile16_offset.x * 16);
            let st = (tile64_occupancy[tile64_index][bit_index / 32] >> (bit_index % 32)) & 3;
            if (st == 2) {
                // empty tile, keep going
                pos.z -= 16;
                continue;
            } else if (st == 1) {
                // Full tile, we can break
                atomicMax(&result[pixel_index], u32(pos.z));
                done = true;
            } else {
                let z16 = u32(pos.z) / 16;
                let tile16_index = tile16.x + tile16.y * size16.x + z16 * size16.x * size16.y;
                for (var k=0; k < 4 && !done; k += 1) {
                    let bit_index = 2 * ((u32(pos.z) % 16) / 4 + tile4_offset.y * 4 + tile4_offset.x * 16);
                    let st = (tile16_occupancy[tile16_index][bit_index / 32] >> (bit_index % 32)) & 3;
                    if (st == 2) {
                        pos.z -= 4;
                        continue;
                    } else if (st == 1) {
                        // Full tile, we can break
                        atomicMax(&result[pixel_index], u32(pos.z));
                        done = true;
                    } else {
                        // Voxel evaluation (or unpopulated, which shouldn't happen?)
                        let out = raycast(tape_start, vec3u(pos));
                        atomicMax(&result[pixel_index], out);
                        if (out > 0) {
                            done = true;
                        }
                        pos.z -= 4;
                    }
                }
            }
        }
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

    // TODO we're only using one result here

    // Unpack the 4x results into pixels
    /*
    for (var i = 0u; i < 4; i += 1u) {
        if (out[i] < 0.0) {
            return pos.z - i;
        }
    }
    return 0u;
    */

    if (out[0] < 0.0) {
        return pos.z;
    } else {
        return 0u;
    }
}
