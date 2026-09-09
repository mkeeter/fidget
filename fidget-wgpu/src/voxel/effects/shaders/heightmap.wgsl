struct HeightmapConfig {
    /// Image size, in voxels
    image_size: vec3u,
    has_color: u32,
}

@group(0) @binding(0) var<uniform> config: HeightmapConfig;

@group(0) @binding(1) var<storage, read> image: array<PackedVoxel>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

/// Produces a heightmap RGBA image
@compute @workgroup_size(8, 8)
fn heightmap_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }

    let i = global_id.x + global_id.y * config.image_size.x;
    let p = image[i];
    // Early exit for unpopulated pixels
    if p.depth == 0u {
        out[i] = 0u;
        return;
    }

    let gp = unpack(p).pixel; // ignoring index for now
    let brightness = clamp(f32(gp.depth) / f32(config.image_size.z), 0.0, 1.0);

    let alpha = 0xFFu << 24;
    if config.has_color != 0 {
        let color = vec4f(unpack4xU8(out[i])) * brightness;
        out[i] = pack4xU8(vec4u(color)) | alpha;
    } else {
        let intensity = u32(brightness * 255);
        out[i] = intensity | (intensity << 8) | (intensity << 16) | alpha;
    }
}
