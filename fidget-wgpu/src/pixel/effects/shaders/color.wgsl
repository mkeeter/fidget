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

@group(1) @binding(0) var<storage, read> distance: array<RawDistancePixel>;
/// Image buffer, which is shape index going in and color going out
@group(1) @binding(1) var<storage, read_write> color: array<u32>;

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

    // Shape indices are in the color buffer; we'll overwrite them here
    var i = global_id.x + config.image_size.x * global_id.y;
    let d = distance[i];

    // Store alpha; early exit if we only care about color for filled pixels
    var alpha = 0u;
    if distance_pixel_is_inside(d) {
        alpha = 0xFF;
    } else if config.only_filled != 0 {
        color[i] = 0x00000000;
        return;
    }

    let tag = color[i];
    if tag >= arrayLength(&shape_start) {
        color[i] = 0xFF0000FF; // corrupt, fill with red
        return;
    }

    // Compute input values
    let m_xy = transformed_inputs(
        Value(f32(global_id.x)),
        Value(f32(global_id.y)),
    );
    let m = array(m_xy[0], m_xy[1], build_imm(config.z));

    let raw_start = shape_start[tag];
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
    color[i] = (alpha << 24) | (u[2] << 16) | (u[1] << 8) | u[0];
}
