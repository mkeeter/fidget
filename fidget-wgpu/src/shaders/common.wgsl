/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,
    _padding1: u32,

    /// Render size, in voxels (always a multiple of 64)
    render_size: vec3u,
    _padding2: u32,

    /// Image size, in voxels
    image_size: vec3u,

    /// Maximum number of workgroups in a single dispatch
    max_tiles_per_dispatch: u32,
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
}

fn nan_f32() -> f32 {
    // Workaround for https://github.com/gpuweb/gpuweb/issues/3749
    let bits = 0xffffffffu;
    return bitcast<f32>(bits);
}

/// Common render configuration (immutable)
@group(0) @binding(0) var<uniform> config: Config;
