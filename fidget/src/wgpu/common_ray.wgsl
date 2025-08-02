/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,
    _padding1: u32,

    /// Image size, in voxels
    image_size: vec3u,
    _padding2: u32,

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
