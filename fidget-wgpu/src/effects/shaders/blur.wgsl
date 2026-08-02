struct BlurConfig {
    /// Image size, in pixels
    image_size: vec2u,

    /// Blur radius, also in pixels
    radius: i32,

    // Padding to 16 bytes
    _pad: u32,
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

    // This is a Kuwahara-style filter: we find a value + score across four
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

    var best = a;
    best = merge(best, b);
    best = merge(best, c);
    best = merge(best, d);

    if best.valid != 0 {
        out[i] = best.mean;
    } else {
        out[i] = image[i];
    }
}

fn merge(best: BlurOutput, other: BlurOutput) -> BlurOutput {
    if best.valid == 0 || (other.valid != 0 && other.score < best.score) {
        return other;
    } else {
        return best;
    }
}

struct BlurOutput {
    mean: f32,
    score: f32,
    valid: u32,
}

fn blur_at(x: i32, y: i32) -> BlurOutput {
    // Find the average value in a square with corner [x, y]
    var sum = 0.0;
    var count = 0.0;
    for (var i = 0; i <= config.radius; i += 1) {
        for (var j = 0; j <= config.radius; j += 1) {
            let tx = x + i;
            let ty = y + j;
            if tx >= 0 && ty >= 0 &&
                u32(tx) < config.image_size.x &&
                u32(ty) < config.image_size.y
            {
                let s = image[u32(tx) + u32(ty) * config.image_size.x];
                if s == s {
                    sum += s;
                    count += 1.0;
                }
            }
        }
    }

    if count == 0.0 {
        return BlurOutput(0.0, 0.0, 0);
    }
    let mean = sum / count;
    var stdev = 0.0;

    // Find the standard deviation of that square patch
    for (var i = 0; i <= config.radius; i += 1) {
        for (var j = 0; j <= config.radius; j += 1) {
            let tx = x + i;
            let ty = y + j;
            if tx >= 0 && ty >= 0 &&
                u32(tx) < config.image_size.x &&
                u32(ty) < config.image_size.y
            {
                let s = image[u32(tx) + u32(ty) * config.image_size.x];
                if s == s {
                    stdev += pow(mean - s, 2);
                }
            }
        }
    }

    return BlurOutput(mean, stdev / count, 1);
}
