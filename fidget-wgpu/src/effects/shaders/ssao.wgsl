struct SsaoConfig {
    /// Image size, in voxels
    image_size: vec3u,
    radius: f32,
}

@group(0) @binding(0) var<uniform> config: SsaoConfig;
@group(0) @binding(1) var<storage, read> image: array<PackedVoxel>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@group(1) @binding(0) var<storage, read> kernel: array<array<f32, 3>>;
@group(1) @binding(1) var<storage, read> noise: array<vec2f>;

@compute @workgroup_size(8, 8)
fn ssao_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }
    let i = global_id.x + global_id.y * config.image_size.x;

    // Early exit for unpopulated pixels
    //
    // Unlike the CPU shader, we use -1.0 as the empty marker (because NaN
    // handling is iffy on the GPU).
    let pixel = unpack(image[i]).pixel;
    if pixel.depth == 0u {
        out[i] = -1.0;
        return;
    }
    let n = pixel.normal;

    // Convert to [-1, 1] coordinate space
    let size_f32 = vec3<f32>(config.image_size);
    let pos_f32 = vec3<f32>(vec2<f32>(global_id.xy) + 0.5, f32(pixel.depth));

    let scale_min = f32(min(
        config.image_size.x,
        min(config.image_size.y, config.image_size.z)
    ));
    let scale_x = scale_min / f32(config.image_size.x);
    let scale_y = scale_min / f32(config.image_size.y);
    let scale_z = scale_min / f32(config.image_size.z);

    // See writeup in `fidget_raster` for details here
    let p = ((pos_f32 / size_f32) - 0.5) * 2.0;
    let rvec = vec3f(noise[pcg2d(global_id.xy).x % arrayLength(&noise)], 0.0);
    let tangent = normalize(rvec - n * dot(rvec, n));
    let bitangent = cross(n, tangent);
    let tbn = mat3x3(tangent, bitangent, n);

    var occlusion = 0.0;
    for (var j=0u; j < arrayLength(&kernel); j++) {
        // offset in world coordinates (with compensation for aspect ratio)
        let k = vec3f(kernel[j][0], kernel[j][1], kernel[j][2]);
        var offset = tbn * k * config.radius;
        offset.x *= scale_x;
        offset.y *= scale_y;
        offset.z *= scale_z;

        // position in world coordinates
        let sample_pos = p + offset;

        // XXX the implementation in `fidget_raster` says "this distorts samples
        // for non-square images"; is this true?
        let pos_voxels = vec3i((sample_pos / 2.0 + 0.5) * vec3f(config.image_size));

        // actual_h is the height from the heightmap image
        var actual_h = 0.0;
        if pos_voxels.x >= 0 &&
           pos_voxels.y >= 0 &&
           u32(pos_voxels.x) < config.image_size.x &&
           u32(pos_voxels.y) < config.image_size.y
        {
            let d = image[
                u32(pos_voxels.x) +
                u32(pos_voxels.y) * config.image_size.x
            ].depth;
            if d != 0 {
                actual_h = f32(d);
            }
        }

        let actual_z = ((f32(actual_h) / f32(config.image_size.z)) - 0.5) * 2.0;
        let dz = sample_pos.z - actual_z;
        if sample_pos.z <= actual_z {
            if dz < config.radius {
                occlusion += 1.0;
            } else if dz < config.radius * 2.0 {
                occlusion += pow((config.radius - (dz - config.radius)) / config.radius, 2);
            }
        }
    }
    out[i] = 1.0 - (occlusion / f32(arrayLength(&kernel)));
}

/// Hash function to mix X and Y integer values
///
/// Source: "Hash Functions for GPU Rendering" (Jarzynski & Olano, 2020)
/// https://jcgt.org/published/0009/03/02/paper.pdf
fn pcg2d(v_in: vec2u) -> vec2u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    return v;
}
