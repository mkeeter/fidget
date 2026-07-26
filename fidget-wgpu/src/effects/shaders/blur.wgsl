struct BlurConfig {
    /// Image size, in pixels
    image_size: vec2u,
    radius: i32,
}

@group(0) @binding(0) var<uniform> config: BlurConfig;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn blur_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Clamp to image size
    if global_id.x >= config.image_size.x ||
       global_id.y >= config.image_size.y
    {
        return;
    }
    let i = global_id.x + global_id.y * config.image_size.x;

    // This is a Kuwahara-style filter: we find a value + store across four
    // quadrants, then pick the best one.
    let a = blur_at(
        i32(global_id.x) - config.radius,
        i32(global_id.y) - config.radius,
    );
    let b = blur_at(
        i32(global_id.x),
        i32(global_id.y) - config.radius,
    );
    let c = blur_at(
        i32(global_id.x) - config.radius,
        i32(global_id.y),
    );
    let d = blur_at(
        i32(global_id.x),
        i32(global_id.y),
    );

}
