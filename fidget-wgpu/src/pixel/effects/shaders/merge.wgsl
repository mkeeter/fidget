struct MergeConfig {
    /// Image size, in pixels
    image_size: vec2u,

    /// Whether or not to denoise when merging (bool)
    denoise: u32,

    /// Offset applied to indices when merging
    index_base: u32,

    /// Number of valid image buffers (0-7)
    image_count: u32,

    // Explicit padding to the nearest multiple of 8
    _pad: u32,
}

@group(0) @binding(0) var<uniform> config: MergeConfig;

@group(0) @binding(1) var<storage, read> image0: array<RawDistancePixel>;
@group(0) @binding(2) var<storage, read> image1: array<RawDistancePixel>;
@group(0) @binding(3) var<storage, read> image2: array<RawDistancePixel>;
@group(0) @binding(4) var<storage, read> image3: array<RawDistancePixel>;
@group(0) @binding(5) var<storage, read> image4: array<RawDistancePixel>;
@group(0) @binding(6) var<storage, read> image5: array<RawDistancePixel>;
@group(0) @binding(7) var<storage, read> image6: array<RawDistancePixel>;
@group(0) @binding(8) var<storage, read_write> out: array<TaggedRawDistancePixel>;


@compute @workgroup_size(8, 8)
fn merge_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y ||
       config.image_count == 0
    {
        return;
    }

    let pos = global_id.xy;
    let i = config.image_size.x * pos.y + pos.x;

    var p = TaggedRawDistancePixel(RawDistancePixel(0), 0); // dummy value
    let b = tag_at(0, pos);
    if config.index_base == 0 {
        p = b;
    } else {
        p = merge_pixel(out[i], b);
    }

    if config.image_count > 1 {
        p = merge_pixel(p, tag_at(1, pos));
    }
    if config.image_count > 2 {
        p = merge_pixel(p, tag_at(2, pos));
    }
    if config.image_count > 3 {
        p = merge_pixel(p, tag_at(3, pos));
    }
    if config.image_count > 4 {
        p = merge_pixel(p, tag_at(4, pos));
    }
    if config.image_count > 5 {
        p = merge_pixel(p, tag_at(5, pos));
    }
    if config.image_count > 6 {
        p = merge_pixel(p, tag_at(6, pos));
    }

    out[i] = p;
}

fn tag_at(image_index: u32, pos: vec2u) -> TaggedRawDistancePixel {
    let i = config.image_size.x * pos.y + pos.x;
    let p = maybe_denoise(image_index, pos);
    return TaggedRawDistancePixel(p, image_index + config.index_base);
}

fn maybe_denoise(image_index: u32, pos: vec2u) -> RawDistancePixel {
    let pixel = read_pixel(image_index, pos.x + pos.y * config.image_size.x);
    if config.denoise != 0 {
        /*
        if pixel.depth > 0 {
            if pixel.normal.z > 0.0 {
                return pixel;
            } else {
                let normal = denoise_at(image_index, pos, pixel);
                return RawDistancePixel(normal, pixel.depth);
            }
        } else {
            return RawDistancePixel(
                vec3f(0.0, 0.0, 0.0),
                0,
            );
        }
        */
        return pixel; /// XXX TODO
    } else {
        return pixel;
    }
}

/*
fn denoise_at(image_index: u32, pos: vec2u, pixel: RawDistancePixel) -> vec3f {
    let empty = RawDistancePixel(vec3f(0.0), 0);
    var data = array<array<GeometryPixel, 3>, 3>(
        array<GeometryPixel, 3>(empty, empty, empty),
        array<GeometryPixel, 3>(empty, pixel, empty),
        array<GeometryPixel, 3>(empty, empty, empty),
    );
    // Populate a 3x3 grid of normals.
    for (var i = -1; i <= 1; i += 1) {
        for (var j = -1; j <= 1; j += 1) {
            let new_pos = vec2i(pos) + vec2i(i, j);
            if (i == 0 && j == 0) ||
                new_pos.x < 0 ||
                new_pos.y < 0 ||
                u32(new_pos.x) >= config.image_size.x ||
                u32(new_pos.y) >= config.image_size.y
            {
                continue;
            }
            data[i + 1][j + 1] = read_pixel(
                image_index,
                u32(new_pos.x) + u32(new_pos.y) * config.image_size.x
            );
        }
    }

    // Iterate over four 2x2 pixel regions, picking the one that's most
    // consistent (most normals agree with mean)
    var scores = array<vec4f, 4>(
        vec4f(0.0),
        vec4f(0.0),
        vec4f(0.0),
        vec4f(0.0),
    );
    for (var i = -1; i <= 0; i += 1) {
        for (var j = -1; j <= 0; j += 1) {
            var sum = vec3f(0.0);
            var count = 0;
            for (var dx = 0; dx <= 1; dx += 1) {
                for (var dy = 0; dy <= 1; dy += 1) {
                    let p = data[i + 1 + dx][j + 1 + dy];
                    if p.depth != 0 && p.normal.z > 0.0 {
                        sum += data[i + 1 + dx][j + 1 + dy].normal;
                        count += 1;
                    }
                }
            }
            if count == 0 {
                continue; // leave score as 0
            }
            var score = 0.0;
            let mean = sum / f32(count);
            for (var dx = 0; dx <= 1; dx += 1) {
                for (var dy = 0; dy <= 1; dy += 1) {
                    if data[i + 1 + dx][j + 1 + dy].depth != 0 {
                        score += dot(mean, data[i + 1 + dx][j + 1 + dy].normal);
                    }
                }
            }
            scores[(i + 1) + (j + 1) * 2] = vec4f(mean, score);
        }
    }

    var best = scores[0];
    for (var i = 0; i < 3; i += 1) {
        if scores[i].w > best.w {
            best = scores[i];
        }
    }
    // Preserve the back-facing normal if we didn't get any valid quadrants
    if best.w == 0.0 {
        return pixel.normal;
    }
    return best.xyz;
}
*/

fn read_pixel(image_index: u32, pixel_index: u32) -> RawDistancePixel {
    switch image_index {
        case 0: { return image0[pixel_index]; }
        case 1: { return image1[pixel_index]; }
        case 2: { return image2[pixel_index]; }
        case 3: { return image3[pixel_index]; }
        case 4: { return image4[pixel_index]; }
        case 5: { return image5[pixel_index]; }
        case 6: { return image6[pixel_index]; }
        default: { return RawDistancePixel(0); }
    }
}

fn merge_pixel(
    a: TaggedRawDistancePixel,
    b: TaggedRawDistancePixel
) -> TaggedRawDistancePixel {
    // haha booleans go brrrrrrr
    let a_inside = distance_pixel_is_inside(a.distance);
    let a_fill = distance_pixel_is_fill(a.distance);
    let b_inside = distance_pixel_is_inside(b.distance);
    let b_fill = distance_pixel_is_fill(b.distance);

    // If either side is inside, then prefer it.
    if a_inside && !b_inside {
        return a;
    } else if b_inside && !a_inside {
        return b;
    }

    // If both sides are fills, prefer A (arbitrarily); otherwise, prefer the
    // non-fill side (since it's a truer value).
    if a_fill && b_fill {
        return a;
    } else if a_fill && !b_fill {
        return b;
    } else if b_fill && !a_fill {
        return a;
    }

    // Otherwise, do a straight distance comparison
    if bitcast<f32>(a.distance.data) < bitcast<f32>(b.distance.data) {
        return a;
    } else {
        return b;
    }
}
