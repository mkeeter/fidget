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
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    // UV coordinates for the quad
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.tex_coords = uv[vertex_index];
    return output;
}

struct GeometryPixel {
    depth: u32,
    normal: vec3<f32>,
}

struct RenderConfig {
    render_mode: u32, // 0 = heightmap, 1 = shaded
    light_direction: vec3<f32>,
    ambient_intensity: f32,
    diffuse_intensity: f32,
}

@group(0) @binding(0) var t_geometry: texture_2d<u32>;
@group(0) @binding(1) var s_geometry: sampler;
@group(0) @binding(2) var<uniform> config: RenderConfig;

// Fragment shader for geometry
@fragment
fn fs_main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let texel = textureLoad(t_geometry, vec2<i32>(tex_coords * vec2<f32>(textureDimensions(t_geometry))), 0);
    
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
    
    // Process based on render mode
    if (config.render_mode == 0u) {
        // Heightmap mode - use grayscale based on depth
        let max_depth = 255.0; // Should be passed in from config or calculated
        let gray = f32(depth) / max_depth;
        return vec4<f32>(gray, gray, gray, 1.0);
    } else {
        // Shaded mode - use normal for lighting
        let norm_normal = normalize(normal);
        
        // Simple Phong-style lighting
        let light_dir = normalize(config.light_direction);
        let diffuse = max(dot(norm_normal, light_dir), 0.0) * config.diffuse_intensity;
        
        // Use normal as a base color and apply lighting
        let norm_color = abs(norm_normal);
        let color = (norm_color * (config.ambient_intensity + diffuse));
        
        return vec4<f32>(color, 1.0);
    }
}