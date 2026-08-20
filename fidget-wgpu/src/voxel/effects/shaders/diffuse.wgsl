struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Image size, in pixels
    image_size: vec2u,

    /// Flexible array member containing start positions (in tape_data)
    shape_start: array<u32>,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> image: array<PackedVoxel>;
@group(0) @binding(2) var<storage, read> tape_data: array<TapeWord>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>; // RGBA

@compute @workgroup_size(8, 8)
fn diffuse_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }

    let i = global_id.x + config.image_size.x * global_id.y;
    let p = unpack(image[i]);
    if p.depth == 0 {
        out[i] = 0x00FFFFFF; // empty, fill with transparent white
    }

    if p.index >= arrayLength(&config.shape_start) {
        out[i] = 0xFF0000FF; // corrupt, fill with red
        return;
    }

    // Compute input values
    let corner_pos = vec3(global_id.x, global_id.y, p.depth);
    let m = transformed_inputs(
        Value(f32(corner_pos.x)),
        Value(f32(corner_pos.y)),
        Value(f32(corner_pos.z)),
    );

    let index = config.shape_start[p.index];
    var stack = Stack(); // dummy value

    // RGB tapes are packed together
    let out_r = run_tape(index, m, &stack)
    let out_g = run_tape(out_r.pos + 1, m, &stack)
    let out_b = run_tape(out_g.pos + 1, m, &stack)

    // Convert to a u32
    let r = u32(clamp(out_r.value.v, 0.0, 1.0) * 255.0);
    let g = u32(clamp(out_g.value.v, 0.0, 1.0) * 255.0);
    let b = u32(clamp(out_b.value.v, 0.0, 1.0) * 255.0);
    out[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
}
