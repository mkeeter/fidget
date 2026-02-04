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

    /// Length of the `tape_data` array (in `u32` words)
    tape_data_capacity: u32,

    /// Image size, in voxels
    image_size: vec3u,

    /// Length of the root tape (plus tile tapes after root tile evaluation)
    ///
    /// `tape_data_offset` should be reset to this value between strata
    root_tape_len: atomic<u32>,

    /// Number of root tiles in a strata
    strata_size: u32,

    // Round to multiple of 8
    _padding: u32,

    /// Tape data, tightly packed per-tile (flexible array member)
    tape_data: array<u32>,
}

/// Dynamic list of tiles, using an atomic bump allocator
struct TileListOutput {
    wg_size: array<atomic<u32>, 3>,
    count: atomic<u32>,

    /// Flexible array member
    active_tiles: array<u32>,
}

/// Read-only version of `TileListOutput`
struct TileListInput {
    wg_size: array<u32, 3>,
    count: u32,
    active_tiles: array<u32>,
}

fn nan_f32() -> f32 {
  // Workaround for https://github.com/gpuweb/gpuweb/issues/3749
  let bits = 0xffffffffu;
  return bitcast<f32>(bits);
}

/// Common render configuration and tape data
@group(0) @binding(0) var<storage, read_write> config: Config;

/// Map from tile to tape index
///
/// See the comment in the computation of `tile_tape_words` for details on how
/// this buffer is packed.
@group(0) @binding(1) var<storage, read_write> tile_tape: array<u32>;

/// For a given position and recursion level, return the offset into `tile_tape`
fn get_tape_offset_for_level(corner_pos: vec3u, level: u32) -> u32 {
    let size64 = config.render_size / 64;
    if level == 64u {
        // 64^3 root tile tapes are densely packed
        let corner_pos64 = corner_pos / 64;
        let index64 = corner_pos64.x
            + corner_pos64.y * size64.x
            + corner_pos64.z * size64.x * size64.y;
        return index64;
    }

    let size16 = config.render_size / 16;
    var offset = size64.x * size64.y * size64.z;
    if level == 16u {
        let corner_pos16 = corner_pos / 16;
        return offset
            + 2 * (corner_pos16.x
                 + corner_pos16.y * size16.x
                 + (corner_pos16.z % (4 * config.strata_size)) * size16.x * size16.y);
    }

    let size4 = config.render_size / 4;
    offset += size16.x
        * size16.y
        * 4  // Z tiles
        * 2; // each item is an (index, z) tuple
    if level == 4u {
        let corner_pos4 = corner_pos / 4;
        return offset
            + 2 * (corner_pos4.x
                 + corner_pos4.y * size4.x
                 + (corner_pos4.z % (16 * config.strata_size)) * size4.x * size4.y);
    }

    return 0;
}

/// For a given voxel position, return the tape start index
///
/// This is the highest-resolution tape index that is valid for the given
/// position, e.g. preferring tapes specialized to 4x4x4 regions, then
/// 16x16x16, then 64x64x64.
fn get_tape_start(corner_pos: vec3u) -> u32 {
    // Presumably the compiler will optimize out common code here
    let index64 = get_tape_offset_for_level(corner_pos, 64u);
    let index16 = get_tape_offset_for_level(corner_pos, 16u);
    let index4 = get_tape_offset_for_level(corner_pos, 4u);

    // The 4^3 and 16^3 tiles are reused between strata, so we have to check
    // that the Z position is valid for the current strata.
    if tile_tape[index4] != 0 && tile_tape[index4 + 1] == corner_pos.z / 4 {
        return tile_tape[index4];
    } else if tile_tape[index16] != 0 && tile_tape[index16 + 1] == corner_pos.z / 16 {
        return tile_tape[index16];
    } else if tile_tape[index64] != 0 {
        return tile_tape[index64];
    } else {
        return 0u;
    }
}
