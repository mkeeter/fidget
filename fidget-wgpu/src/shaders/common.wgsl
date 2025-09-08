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

struct TapeIndex {
    addr: u32,
    index: u32,
}

/// For a given position, return the tape start index
fn get_tape_start(corner_pos: vec3u) -> TapeIndex {
    // 64^3 root tile tapes are densely packed
    let size64 = config.render_size / 64;
    let corner_pos64 = corner_pos / 64;
    let index64 = corner_pos64.x
        + corner_pos64.y * size64.x
        + corner_pos64.z * size64.x * size64.y;
    let tape_index64 = tile_tape[index64];

    // We use the high bit to indicate a recursive address
    if (tape_index64 & (1u << 31)) == 0 {
        return TapeIndex(index64, tape_index64);
    }

    // Otherwise, we have to recurse!  Let's find the relative offset of the
    // 16^3 tile within the root tile.
    let corner_pos16 = (corner_pos % 64u) / 16; // 0-3 on each axis
    let index16 = corner_pos16.x
        + corner_pos16.y * 4u
        + corner_pos16.z * 16u;

    // Look up the 16^3 tile tape
    let addr16 = (tape_index64 & 0x7FFFFFFF) + index16;
    let tape_index16 = tile_tape[addr16];
    if (tape_index16 & (1u << 31)) == 0 {
        return TapeIndex(addr16, tape_index16);
    }

    // Find the relative offset of the 4^3 tile within the 16^3 tile
    let corner_pos4 = (corner_pos % 16u) / 4; // 0-3 on each axis
    let index4 = corner_pos4.x
        + corner_pos4.y * 4u
        + corner_pos4.z * 16u;

    // This is the end of the tree!
    let addr4 = (tape_index16 & 0x7FFFFFFF) + index4;
    return TapeIndex(addr4, tile_tape[addr4]);
}
