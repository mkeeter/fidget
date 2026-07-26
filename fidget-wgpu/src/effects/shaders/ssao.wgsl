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
    let pixel = unpack(image[i]).pixel;
    if pixel.depth == 0u {
        out[i] = nan_f32();
        return;
    }
    let n = pixel.normal;

    // Convert to [-1, 1] coordinate space
    let size_f32 = vec3<f32>(config.image_size);
    let pos_f32 = vec3<f32>(vec2<f32>(global_id.xy), f32(pixel.depth));

    // See writeup in `fidget_raster` for details here
    let p = ((pos_f32 / size_f32) - 0.5) * 2.0;
    let rvec = vec3f(noise[(i * 19) % arrayLength(&noise)], 0.0);
    let tangent = normalize(rvec - n * dot(rvec, n));
    let bitangent = cross(n, tangent);
    let tbn = mat3x3(tangent, bitangent, n);

    var occlusion = 0.0;
    for (var j=0u; j < arrayLength(&kernel); j++) {
        let k = vec3f(kernel[j][0], kernel[j][1], kernel[j][2]);
        let sample_pos = tbn * k * config.radius + p;

        // XXX the implementation in `fidget_raster` says "this distorts samples
        // for non-square images"; is this true?
        let pos_voxels = vec3i((sample_pos / 2.0 + 0.5) * vec3f(config.image_size));

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
            if d == 0 {
                continue;
            }
            actual_h = f32(d);
        } else {
            continue;
        }

        let actual_z = ((f32(actual_h) / f32(config.image_size.z)) - 0.5) * 2.0;
        let dz = sample_pos.z - actual_z;
        if dz < config.radius {
            if sample_pos.z <= actual_z {
                occlusion += 1.0;
            }
        } else if dz < config.radius * 2.0 && sample_pos.z <= actual_z {
            occlusion += pow((config.radius - (dz - config.radius)) / config.radius, 2);
        }
    }
    out[i] = 1.0 - (occlusion / f32(arrayLength(&kernel)));
}
