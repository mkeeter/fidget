/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Next empty position in `tape_data`
    tape_data_offset: atomic<u32>,

    /// Render size, in voxels (always a multiple of 64)
    render_size: vec3u,

    // Next empty position in `tile_tapes`
    tile_tapes_offset: atomic<u32>,

    /// Image size, in voxels
    image_size: vec3u,

    /// Length of the `tape_data` array (in `u32` words)
    tape_data_capacity: u32,

    /// Length of the root tape (plus tile tapes after root tile evaluation)
    ///
    /// `tape_data_offset` should be reset to this value between strata
    root_tape_len: atomic<u32>,

    /// Tape data, tightly packed per-tile
    tape_data: array<u32>,
}

struct TileListOutput {
    wg_size: array<atomic<u32>, 3>,
    count: atomic<u32>,
    active_tiles: array<u32>,
}

struct TileListInput {
    wg_size: array<u32, 3>,
    count: u32,
    active_tiles: array<u32>,
}

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7FC00000);
}

/// Common render configuration and tape data
@group(0) @binding(0) var<storage, read_write> config: Config;

/// Tape tree (with offset given by config.tile_tapes_offset)
@group(0) @binding(1) var<storage, read_write> tile_tape: array<u32>;

/// For a given position, return the tape start index
fn get_tape_start(corner_pos: vec3u) -> u32 {
    let size64 = config.render_size / 64;
    let corner_pos64 = corner_pos / 64;
    let index64 = corner_pos64.x
        + corner_pos64.y * size64.x
        + corner_pos64.z * size64.x * size64.y;
    let tape_index = tile_tape[index64];
    return tape_index; // TODO recursion
}
