struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

// Vertex shader to render a full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Create a "full screen quad" from just the vertex_index
    // Maps vertex indices (0,1,2,3) to positions:
    // (-1,1)----(1,1)
    //   |         |
    // (-1,-1)---(1,-1)
    const POS = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    // UV coordinates for the quad
    const UV = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(POS[vertex_index], 0.0, 1.0);
    output.tex_coords = UV[vertex_index];
    return output;
}

struct GeometryPixel {
    depth: u32,
    normal: vec3<f32>,
}

struct Light {
    position: vec3<f32>,
    intensity: f32,
}

struct RenderConfig {
    render_mode: u32, // 0 = heightmap, 1 = shaded
    max_depth: u32,
}

@group(0) @binding(0) var t_geometry: texture_2d<u32>;
@group(0) @binding(1) var s_geometry: sampler;
@group(0) @binding(2) var<uniform> config: RenderConfig;

// Fragment shader for geometry
@fragment
fn fs_main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let texel = textureLoad(
        t_geometry,
        vec2<i32>(tex_coords * vec2<f32>(textureDimensions(t_geometry))),
        0);

    // First 32 bits are depth, next 32 bits are normal.x, etc.
    let depth = texel.x;

    // If depth is 0, this pixel is transparent
    if (depth == 0u) {
        discard;
    }

    // Extract normal (stored as bits in texel.yzw)
    let normal_x = bitcast<f32>(texel.y);
    let normal_y = bitcast<f32>(texel.z);
    let normal_z = bitcast<f32>(texel.w);
    let normal = vec3<f32>(normal_x, normal_y, normal_z);

    if (config.render_mode == 0u) {
        // Heightmap mode - use grayscale based on depth
        let gray = f32(depth) / f32(config.max_depth);
        return vec4<f32>(gray, gray, gray, 1.0);
    } else if (config.render_mode == 1u) {
        // RGB mode, using normals
        let norm_normal = normalize(normal);
        return vec4<f32>(abs(norm_normal), 1.0);
    } else {
        // Shaded mode
        let p = vec3<f32>(
            (tex_coords.xy - 0.5) * 2.0,
            2.0 * (f32(depth) / f32(config.max_depth) - 0.5)
        );
        let n = normalize(normal);
        const LIGHTS = array<Light, 3>(
            Light(vec3<f32>(5.0, -5.0, 10.0), 0.5),
            Light(vec3<f32>(-5.0, 0.0, 10.0), 0.15),
            Light(vec3<f32>(0.0, -5.0, 10.0), 0.15)
        );
        var accum: f32 = 0.2;
        for (var i = 0u; i < 3u; i = i + 1u) {
            let light = LIGHTS[i];
            let light_dir = normalize(light.position - p);
            accum = accum + max(dot(light_dir, n), 0.0) * light.intensity;
        }
        accum = clamp(accum, 0.0, 1.0);
        return vec4<f32>(accum, accum, accum, 1.0);
    }
}
