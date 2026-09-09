//! Pass to compute per-pixel color based on pixel indices

struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat4x4f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    // Explicit padding
    _pad: u32,

    /// Image size, in pixels
    image_size: vec2u,

    /// Tape data, tightly packed per-tile (flexible array member)
    tape_data: array<TapeWord>,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read> shape_start: array<u32>;

/// Array of values for (non-xyz) variables
@group(0) @binding(2) var<storage, read> var_values: array<f32>;

@group(1) @binding(0) var<storage, read> image: array<PackedVoxel>;
@group(1) @binding(1) var<storage, read_write> color: array<u32>; // RGBA

@compute @workgroup_size(8, 8)
fn color_main(
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
    if p.pixel.depth == 0 {
        color[i] = 0x00FFFFFF; // empty, fill with transparent white
        return;
    }

    if p.index >= arrayLength(&shape_start) {
        color[i] = 0xFF0000FF; // corrupt, fill with red
        return;
    }

    // Compute input values
    let corner_pos = vec3(global_id.x, global_id.y, p.pixel.depth);
    let m = transformed_inputs(
        Value(f32(corner_pos.x)),
        Value(f32(corner_pos.y)),
        Value(f32(corner_pos.z)),
    );

    let raw_start = shape_start[p.index];
    let index = raw_start & 0x7FFFFFFF;
    let is_hsl = (raw_start & (1u << 31)) != 0;
    var stack = Stack(); // dummy value

    // Color channel tapes are packed together
    let out_0 = run_tape(index, m, &stack);
    let out_1 = run_tape(out_0.pos, m, &stack);
    let out_2 = run_tape(out_1.pos, m, &stack);
    let out = vec3f(
        clamp(out_0.value.v, 0.0, 1.0),
        clamp(out_1.value.v, 0.0, 1.0),
        clamp(out_2.value.v, 0.0, 1.0)
    );

    var channels: vec3f;
    if is_hsl {
        channels = hsl_to_rgb(out);
    } else {
        channels = out;
    }

    let u = vec3u(channels * 255.0);
    color[i] = (0xFFu << 24) | (u[2] << 16) | (u[1] << 8) | u[0];
}
