/// Backfill tile_zmin from subtile_zmin
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> image_heightmap: array<u32>;
@group(0) @binding(2) var<storage, read_write> image_out: array<vec4u>;

@compute @workgroup_size(8, 8)
fn normals_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Out of bounds, return
    if (global_id.x >= config.image_size.x ||
        global_id.y >= config.image_size.y)
    {
        return;
    }

    let pixel_index_xy = global_id.x + global_id.y * config.image_size.x;
    let z = image_heightmap[pixel_index_xy];
    if (z == 0u) {
        image_out[pixel_index_xy] = vec4u(0u);
    }

    // Store gradients with dx, dy, dz in xyz and value in w
    let gx = vec4f(1.0, 0.0, 0.0, f32(global_id.x));
    let gy = vec4f(0.0, 1.0, 0.0, f32(global_id.y));
    let gz = vec4f(0.0, 0.0, 1.0, f32(z));

    var ts = mat4x4f(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    for (var i = 0; i < 4; i++) {
        ts[i] = op_mul(build_imm(config.mat[0][i]), gx)
            + op_mul(build_imm(config.mat[1][i]), gy)
            + op_mul(build_imm(config.mat[2][i]), gz)
            + build_imm(config.mat[3][i]);
    }
    // Build up input map
    var m = array(vec4f(0.0), vec4f(0.0), vec4f(0.0));
    if (config.axes.x < 3) {
        m[config.axes.x] = op_div(ts[0], ts[3]);
    }
    if (config.axes.y < 3) {
        m[config.axes.y] = op_div(ts[1], ts[3]);
    }
    if (config.axes.z < 3) {
        m[config.axes.z] = op_div(ts[2], ts[3]);
    }

    let out = run_tape(m)[0];
    image_out[pixel_index_xy] = vec4u(
        z,
        bitcast<u32>(out.x),
        bitcast<u32>(out.y),
        bitcast<u32>(out.z)
    );
}

fn op_neg(lhs: vec4f) -> vec4f {
    return -lhs;
}

fn op_abs(lhs: vec4f) -> vec4f {
    if (lhs.w < 0.0) {
        return -lhs;
    } else {
        return lhs;
    }
}

fn op_recip(lhs: vec4f) -> vec4f {
    let v2 = -lhs.w * lhs.w;
    return vec4f(
        lhs.xyz / v2,
        1.0 / lhs.w,
    );
}

fn op_sqrt(lhs: vec4f) -> vec4f {
    let v = sqrt(lhs.w);
    return vec4f(lhs.xyz / (2.0 * v), v);
}

fn op_floor(lhs: vec4f) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, floor(lhs.w));
}

fn op_ceil(lhs: vec4f) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, ceil(lhs.w));
}

fn op_round(lhs: vec4f) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, round(lhs.w));
}

fn op_square(lhs: vec4f) -> vec4f {
    return op_mul(lhs, lhs);
}

fn op_sin(lhs: vec4f) -> vec4f {
    let c = cos(lhs.w);
    return vec4f(lhs.xyz * c, sin(lhs.w));
}

fn op_cos(lhs: vec4f) -> vec4f {
    let s = -sin(lhs.w);
    return vec4f(lhs.xyz * s, cos(lhs.w));
}

fn op_tan(lhs: vec4f) -> vec4f {
    let c = cos(lhs.w);
    let c2 = c * c;
    return vec4f(lhs.xyz / c, tan(lhs.w));
}

fn op_asin(lhs: vec4f) -> vec4f {
    let r = sqrt(1.0 - lhs.w * lhs.w);
    return vec4f(lhs.xyz / r, asin(lhs.w));
}

fn op_acos(lhs: vec4f) -> vec4f {
    let r = sqrt(1.0 - lhs.w * lhs.w);
    return vec4f(-lhs.xyz / r, acos(lhs.w));
}

fn op_atan(lhs: vec4f) -> vec4f {
    let r = lhs.w * lhs.w + 1.0;
    return vec4f(lhs.xyz / r, atan(lhs.w));
}

fn op_exp(lhs: vec4f) -> vec4f {
    let v = exp(lhs.w);
    return vec4f(lhs.xyz * v, v);
}

fn op_log(lhs: vec4f) -> vec4f {
    let v = log(lhs.w);
    return vec4f(lhs.xyz / v, v);
}

fn op_not(lhs: vec4f) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, f32(lhs.w == 0.0));
}

fn op_compare(lhs: vec4f, rhs: vec4f) -> vec4f {
    if (lhs.w != lhs.w || rhs.w != rhs.w) {
        return vec4f(0.0, 0.0, 0.0, nan_f32());
    } else if (lhs.w < rhs.w) {
        return vec4f(0.0, 0.0, 0.0, -1.0);
    } else if (lhs.w > rhs.w) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    } else {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
}

fn op_and(lhs: vec4f, rhs: vec4f) -> vec4f {
    if (lhs.w != 0.0) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: vec4f, rhs: vec4f) -> vec4f {
    if (lhs.w == 0.0) {
        return lhs;
    } else {
        return rhs;
    }
}

fn build_imm(v: f32) -> vec4f {
    return vec4f(0.0, 0.0, 0.0, v);
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if (r < 0.0) {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn div_euclid(lhs: f32, rhs: f32) -> f32 {
    let q = trunc(lhs / rhs);
    if (lhs % rhs < 0.0) {
        if (rhs > 0.0) {
            return q - 1.0;
        } else {
            return q + 1.0;
        }
    } else {
        return q;
    }
}

fn op_mod(lhs: vec4f, rhs: vec4f) -> vec4f {
    let e = div_euclid(lhs.w, rhs.w);
    return vec4f(
        lhs.xyz - rhs.xyz * e,
        rem_euclid(lhs.w, rhs.w)
    );
}

fn op_add(lhs: vec4f, rhs: vec4f) -> vec4f {
    return lhs + rhs;
}

fn op_sub(lhs: vec4f, rhs: vec4f) -> vec4f {
    return lhs - rhs;
}

fn op_min(lhs: vec4f, rhs: vec4f) -> vec4f {
    if (lhs.w < rhs.w) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_max(lhs: vec4f, rhs: vec4f) -> vec4f {
    if (lhs.w > rhs.w) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_mul(lhs: vec4f, rhs: vec4f) -> vec4f {
    return vec4f(
        lhs.xyz * rhs.w + rhs.xyz * lhs.w,
        lhs.w * rhs.w
    );
}

fn op_div(lhs: vec4f, rhs: vec4f) -> vec4f {
    let d = rhs.w * rhs.w;
    return vec4f(
        (rhs.w * lhs.xyz - lhs.w * rhs.xyz) / d,
        lhs.w / rhs.w
    );
}

fn op_atan2(lhs: vec4f, rhs: vec4f) -> vec4f {
    let d = rhs.w * rhs.w + lhs.w * lhs.w;
    return vec4f(
        (rhs.w * lhs.xyz - lhs.w * rhs.xyz) / d,
        atan2(lhs.w, rhs.w)
    );
}
