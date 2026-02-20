/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Render size, in voxels (always a multiple of 64)
    render_size: vec3u,

    /// Image size, in voxels
    image_size: vec3u,
}

/// Dynamic list of tiles, using an atomic bump allocator
struct TileListOutput {
    /// Bump allocator
    count: atomic<u32>,

    /// Flexible array member, must be sized to fit maximum tile count
    tiles: array<ActiveTile>,
}

/// Read-only version of `TileListOutput`
struct TileListInput {
    count: u32,
    active_tiles: array<ActiveTile>,
}

struct TapeData {
    /// Offset of the first free word in `data`
    ///
    /// This must be initialized based on tape length
    offset: atomic<u32>,

    /// Total capacity of `data` (in words)
    capacity: u32,

    /// Flexible array member of tape data
    ///
    /// The first valid tape (at index 0) must be the root tape
    data: array<u32>,
}

/// Single voxel, as a packed `(Z, tape index)` tuple
///
/// The Z value occupies the upper 12 bits; the tape index is the lower 20.
struct Voxel {
    value: atomic<u32>,
}

/// Tile to be evaluated
struct ActiveTile {
    /// Tile position, with x/y/z values packed into a single `u32`
    tile: u32,
    /// Start of this tile's tape in the tape data array
    tape_index: u32,
}

/// Indirect dispatch plan for a round of interval tile dispatch
struct Dispatch {
    /// Indirect dispatch size
    wg_dispatch: vec3u,

    /// Number of tiles actually in this dispatch
    ///
    /// Note that `wg_dispatch` may dispatch fewer workgroups than `tile_count`
    /// when used in the voxel shader, because that shader loops over tiles.
    tile_count: u32,
}

fn nan_f32() -> f32 {
    // Workaround for https://github.com/gpuweb/gpuweb/issues/3749
    let bits = 0xffffffffu;
    return bitcast<f32>(bits);
}

/// Common render configuration (immutable)
@group(0) @binding(0) var<uniform> config: Config;
