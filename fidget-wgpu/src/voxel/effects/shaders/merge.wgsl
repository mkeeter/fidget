struct MergeConfig {
    /// Image size, in pixels
    image_size: vec2u,

    /// Whether or not to denoise when merging (bool)
    denoise: u32,

    /// Offset applied to indices when merging
    index_base: u32,
}

@group(0) @binding(0) var<uniform> config: MergeConfig;
@group(0) @binding(1) var<storage, read> image: array<GeometryPixel>;
@group(0) @binding(2) var<storage, read_write> out: array<PackedVoxel>;


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

    // Read and pack the pixel from the input image
    let b = pack_at(pos);

    // Either write the pixel directly or merge it with the previous value
    if config.index_base == 0 {
        out[i] = b;
    } else {
        out[i] = merge_pixel(out[i], b);
    }
}

fn pack_at(pos: vec2u) -> PackedVoxel {
    let i = config.image_size.x * pos.y + pos.x;
    let p = maybe_denoise(pos);
    return pack(TaggedGeometryPixel(p, config.index_base));
}

fn maybe_denoise(pos: vec2u) -> GeometryPixel {
    let pixel = image[pos.x + pos.y * config.image_size.x];
    if config.denoise != 0 {
        if pixel.depth > 0 {
            if pixel.normal.z > 0.0 {
                return pixel;
            } else {
                let normal = denoise_at(pos, pixel);
                return GeometryPixel(normal, pixel.depth);
            }
        } else {
            return GeometryPixel(
                vec3f(0.0, 0.0, 0.0),
                0,
            );
        }
    } else {
        return pixel;
    }
}

fn denoise_at(pos: vec2u, pixel: GeometryPixel) -> vec3f {
    let empty = GeometryPixel(vec3f(0.0), 0);
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
            data[i + 1][j + 1] = image[
                u32(new_pos.x) + u32(new_pos.y) * config.image_size.x
            ];
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

fn merge_pixel(a: PackedVoxel, b: PackedVoxel) -> PackedVoxel {
    if a.depth >= b.depth {
        return a;
    } else {
        return b;
    }
}
