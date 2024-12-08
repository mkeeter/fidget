/// Render configuration
@group(0) @binding(0) var<uniform> config: Config;

/// Tiles to render
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;

/// Input tape(s), serialized to bytecode
@group(0) @binding(2) var<storage, read> tape: array<u32>;

/// Output array (single values), of same length as `vars`
@group(0) @binding(3) var<storage, read_write> result: array<atomic<u32>>;

/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Tile size, in pixels
    tile_size: u32,

    /// Window size, in pixels
    window_size: vec2u,

    /// Number of tiles to render
    tile_count: u32,

    /// Explicit padding
    _padding: u32,
}

/// Per-tile render configuration
struct Tile {
    /// Corner of the tile, as a voxel position
    corner: vec3u,

    /// Starting point of this tile in the `tape` array
    start: u32,
}
