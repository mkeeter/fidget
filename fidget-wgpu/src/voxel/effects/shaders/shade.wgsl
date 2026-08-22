struct ShadeConfig {
    /// Image size, in voxels
    image_size: vec3u,

    /// Flags for the validity of `occlusion` and output color
    flags: u32,
}

const SHADE_CONFIG_HAS_SSAO: u32 = 1;
const SHADE_CONFIG_HAS_COLOR: u32 = 2;

struct Light {
    position: vec3<f32>,
    intensity: f32,
}

@group(0) @binding(0) var<uniform> config: ShadeConfig;

@group(0) @binding(1) var<storage, read> image: array<PackedVoxel>;
@group(0) @binding(2) var<storage, read> occlusion: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

/// Applies shading to get an RGBA image
@compute @workgroup_size(8, 8)
fn shade_main(
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

    // Convert to [-1, 1] coordinate space
    let size_f32 = vec3<f32>(config.image_size);
    let pos_f32 = vec3<f32>(vec2<f32>(global_id.xy), f32(p.depth));
    let pos = ((pos_f32 / size_f32) - 0.5) * 2.0;

    const LIGHTS = array<Light, 3>(
        Light(vec3<f32>(5.0, -5.0, 10.0), 0.5),
        Light(vec3<f32>(-5.0, 0.0, 10.0), 0.15),
        Light(vec3<f32>(0.0, -5.0, 10.0), 0.15)
    );
    var accum: f32 = 0.2;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let light = LIGHTS[i];
        let light_dir = normalize(light.position - pos);
        accum = accum + max(dot(light_dir, gp.normal), 0.0) * light.intensity;
    }
    if (config.flags & SHADE_CONFIG_HAS_SSAO) != 0 {
        accum *= occlusion[i] * 0.6 + 0.4;
    }
    let brightness = clamp(accum, 0.0, 1.0);
    let alpha = 0xFF << 24;
    if (config.flags & SHADE_CONFIG_HAS_COLOR) != 0 {
        let color = vec4f(unpack4xU8(out[i])) * brightness;
        out[i] = pack4xU8(vec4u(color)) | alpha;
    } else {
        let intensity = u32(brightness * 255);
        out[i] = intensity | (intensity << 8) | (intensity << 16);
    }
}
