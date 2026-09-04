struct MergeConfig {
    /// Image size, in pixels
    image_size: vec2u,

    /// Whether or not to remove NaNs when merging images
    remove_nans: u32,

    /// Offset applied to indices when merging
    index_base: u32,
}

@group(0) @binding(0) var<uniform> config: MergeConfig;

@group(0) @binding(1) var<storage, read> image: array<RawDistancePixel>;

// Distance and index values, packed as separate images
@group(0) @binding(2) var<storage, read_write> out: array<u32>;


@compute @workgroup_size(8, 8)
fn merge_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }

    let pos = global_id.xy;
    let i = config.image_size.x * pos.y + pos.x;
    let offset = config.image_size.x * config.image_size.y;

    var p = TaggedRawDistancePixel(RawDistancePixel(0), 0); // dummy value
    let b = tag_at(pos);
    if config.index_base == 0 {
        p = b;
    } else {
        p = TaggedRawDistancePixel(RawDistancePixel(out[i]), out[i + offset]);
        p = merge_pixel(p, b);
    }

    out[i] = p.distance.data;
    out[i + offset] = p.index;
}

fn tag_at(pos: vec2u) -> TaggedRawDistancePixel {
    let i = config.image_size.x * pos.y + pos.x;
    let p = maybe_remove_nans(pos);
    return TaggedRawDistancePixel(p, config.index_base);
}

fn maybe_remove_nans( pos: vec2u) -> RawDistancePixel {
    let pixel = image[pos.x + pos.y * config.image_size.x];
    if config.remove_nans != 0 && distance_pixel_is_fill(pixel) {
        return remove_nans_at( pos, pixel);
    } else {
        return pixel;
    }
}

// Replace fill pixels (NaN-boxed) with the average of their actual-distance
// neighbors, falling back to infinity if that fails.  This prevents glitchiness
// on the edges of models: If a NaN-boxed fill pixel is exactly at the edge of a
// model, linear interpolation in the texture means that every pixel
// interpolated with the infinite pixel is also NaN.
fn remove_nans_at(
    pos: vec2u,
    pixel: RawDistancePixel) -> RawDistancePixel
{
    var inside_count = 0.0;
    var inside_avg = 0.0;
    var outside_count = 0.0;
    var outside_avg = 0.0;

    for (var dy = -1i; dy <= 1; dy += 1) {
        let y = i32(pos.y) + dy;
        if y < 0 || u32(y) >= config.image_size.y {
            continue;
        }
        for (var dx = -1i; dx <= 1; dx += 1) {
            let x = i32(pos.x) + dx;
            if x < 0 || u32(x) >= config.image_size.x {
                continue;
            }

            let p = image[u32(x) + u32(y) * config.image_size.x];

            if !distance_pixel_is_fill(p) {
                let d = bitcast<f32>(p.data);
                if d < 0.0 {
                    inside_avg += d;
                    inside_count += 1;
                } else if d > 0.0 {
                    outside_avg += d;
                    outside_count += 1;
                }
            }
        }
    }

    let pixel_is_inside = distance_pixel_is_inside(pixel);
    if pixel_is_inside && inside_count > 0 {
        return distance_pixel_value(inside_avg / inside_count);
    } else if !pixel_is_inside && outside_count > 0 {
        return distance_pixel_value(outside_avg / outside_count);
    } else if inside_count + outside_count > 0 {
        let avg = (inside_avg + outside_avg) / (inside_count + outside_count);
        if (avg < 0.0) == pixel_is_inside {
            return distance_pixel_value(avg);
        }
    }
    // Fallback: set to ±infinity
    if pixel_is_inside {
        return RawDistancePixel(0xFF800000); // -infinity
    } else {
        return RawDistancePixel(0x7F800000); // +infinity
    }
}

fn merge_pixel(
    a: TaggedRawDistancePixel,
    b: TaggedRawDistancePixel
) -> TaggedRawDistancePixel {
    // The rule is simple: prefer the inside side, or `b` if they have the same
    // sign.  This is less accurate than `min` if both values have a true
    // distance, but produces a consistent result even with fills.
    let a_inside = distance_pixel_is_inside(a.distance);
    let b_inside = distance_pixel_is_inside(b.distance);
    if a_inside && !b_inside {
        return a;
    } else {
        return b;
    }
}

/// Distance pixel with an associated shape index
struct TaggedRawDistancePixel {
    /// Distance associated with this pixel
    distance: RawDistancePixel,

    /// Shape index
    index: u32,
}
