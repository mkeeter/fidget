/// Equivalent to RawDistancePixel in Rust, but storing a u32 instead of f32
/// (because shaders are bad about handling NaN values)
struct RawDistancePixel {
    data: u32,
}

// NAN with characteristic bit pattern in mantissa
const DISTANCE_PIXEL_KEY: u32 = 0x7fc1ec00;

fn distance_pixel_fill(depth: u32, inside: bool) -> RawDistancePixel {
    let data = (depth << 1) | u32(inside) | DISTANCE_PIXEL_KEY;
    return RawDistancePixel(data);
}

fn distance_pixel_is_fill(d: RawDistancePixel) -> bool {
    return (d.data & DISTANCE_PIXEL_KEY) == DISTANCE_PIXEL_KEY;
}

fn distance_pixel_is_inside(d: RawDistancePixel) -> bool {
    if distance_pixel_is_fill(d) {
        return (d.data & 1u) != 0;
    } else {
        return bitcast<f32>(d.data) < 0.0;
    }
}

fn distance_pixel_value(v: f32) -> RawDistancePixel {
    if v != v {
        // attempt to canonicalize NaN values, in case someone is trying to do
        // something weird (???)
        return RawDistancePixel(bitcast<u32>(nan_f32()));
    } else {
        return RawDistancePixel(bitcast<u32>(v));
    }
}
