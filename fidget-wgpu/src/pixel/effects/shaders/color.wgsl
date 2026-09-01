//! Pass to compute per-pixel color based on pixel indices

struct Config {
    /// Screen-to-model transform matrix, converting pixels to model space
    mat: mat3x3f,

    /// Mapping from X, Y, Z to input indices
    axes: vec3u,

    // Z height at which to evaluate
    z: f32,

    /// Image size, in pixels
    image_size: vec2u,

    /// Only compute color for filled pixels when this is non-zero
    only_filled: u32,

    // manual alignment
    _pad: array<u32, 3>,

    /// Tape data, tightly packed per-tile (flexible array member)
    tape_data: array<TapeWord>,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read> shape_start: array<u32>;

/// Array of values for (non-xyz) variables
@group(0) @binding(2) var<storage, read> var_values: array<f32>;

/// Image buffer, which is shape index going in and color going out
@group(1) @binding(0) var<storage, read_write> image: array<u32>;

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

    // Shape indices are in the second half of the buffer, offset by image size
    var i = global_id.x + config.image_size.x * global_id.y;
    let distance = RawDistancePixel(image[i]);

    // Shift to address shape index / color
    i += config.image_size.x * config.image_size.y;

    // Store alpha; early exit if we only care about color for filled pixels
    var alpha = 0u;
    if distance_pixel_is_inside(distance) {
        alpha = 0xFF;
    } else if config.only_filled != 0 {
        image[i] = 0x00000000;
        return;
    }

    let tag = image[i];
    if tag >= arrayLength(&shape_start) {
        image[i] = 0xFF0000FF; // corrupt, fill with red
        return;
    }

    // Compute input values
    let m_xy = transformed_inputs(
        Value(f32(global_id.x)),
        Value(f32(global_id.y)),
    );
    let m = array(m_xy[0], m_xy[1], build_imm(config.z));

    let index = shape_start[tag];
    var stack = Stack(); // dummy value

    // RGB tapes are packed together
    let out_r = run_tape(index, m, &stack);
    let out_g = run_tape(out_r.pos, m, &stack);
    let out_b = run_tape(out_g.pos, m, &stack);

    // Convert to a u32
    let r = u32(clamp(out_r.value.v, 0.0, 1.0) * 255.0);
    let g = u32(clamp(out_g.value.v, 0.0, 1.0) * 255.0);
    let b = u32(clamp(out_b.value.v, 0.0, 1.0) * 255.0);

    image[i] = (alpha << 24) | (b << 16) | (g << 8) | r;
}
