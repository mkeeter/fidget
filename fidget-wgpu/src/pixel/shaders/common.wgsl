/// Global render configuration
///
/// Variables are ordered to require no padding
struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat3x3f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    /// Next empty position in `tape_data`
    tape_data_offset: atomic<u32>,

    /// Render size, in voxels (always a multiple of 64)
    render_size: vec2u,

    /// Image size, in pixels
    image_size: vec2u,

    /// Z position at which to render the image
    z: f32,

    /// Flag indicating whether to recurse down to individual pixels
    pixel_perfect: u32,

    /// Length of the `tape_data` array (in `u32` words)
    tape_data_capacity: u32,

    /// Padding
    _pad: u32,

    /// Tape data, tightly packed per-tile (flexible array member)
    tape_data: array<TapeWord>,
}

/// Common render configuration and tape data
@group(0) @binding(0) var<storage, read_write> config: Config;

/// Map from tile to tape index
///
/// See the comment in the computation of `tile_tape_words` for details on how
/// this buffer is packed.
@group(0) @binding(1) var<storage, read_write> tile_tape: array<u32>;

/// Array of values for (non-xyz) variables
@group(1) @binding(0) var<storage, read> var_values: array<f32>;

/// For a given position and recursion level, return the offset into `tile_tape`
fn get_tape_offset_for_level(corner_pos: vec2u, level: u32) -> u32 {
    var offset = 0u; // current position in the buffer

    let size64 = config.render_size / 64;
    if level == 64u {
        // 64^2 root tile tapes are densely packed
        let corner_pos64 = corner_pos / 64;
        let index64 = corner_pos64.x + corner_pos64.y * size64.x;
        return index64;
    }
    offset += size64.x * size64.y;

    let size8 = config.render_size / 8;
    if level == 8u {
        let corner_pos8 = corner_pos / 8;
        return offset + corner_pos8.x + corner_pos8.y * size8.x;
    }

    // invalid level, return the base tape
    return 0;
}

/// Equivalent to RawDistancePixel in Rust, but storing a u32 instead of f32
/// (because shaders are bad about handling NaN values)
struct RawDistancePixel {
    data: u32,
}

const DISTANCE_PIXEL_KEY: u32 = 0xF6 << 9;

fn distance_pixel_fill(depth: u32, inside: bool) -> RawDistancePixel {
    let data = 0x7FC00000u | (depth << 1) | u32(inside) | DISTANCE_PIXEL_KEY;
    return RawDistancePixel(data);
}

fn distance_pixel_value(v: f32) -> RawDistancePixel {
    // we don't canonicalize float NaNs here; maybe we should?
    return RawDistancePixel(bitcast<u32>(v));
}
