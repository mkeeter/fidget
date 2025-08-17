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
    _padding3: u32,

    /// Tape data, tightly packed per-tile
    tape_data: array<u32>,
}

struct TileListOutput {
    wg_size: array<atomic<u32>, 3>,
    count: atomic<u32>,
    size: u32,
    active_tiles: array<u32>,
}

struct TileListInput {
    wg_size: array<u32, 3>,
    count: u32,
    size: u32,
    active_tiles: array<u32>,
}

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7FC00000);
}

/// Common render configuration and tape data
@group(0) @binding(0) var<storage, read> config: Config;

/// Tape start position, on a per-tile basis (densely packed)
@group(0) @binding(1) var<storage, read> tile_tape: array<u32>;
