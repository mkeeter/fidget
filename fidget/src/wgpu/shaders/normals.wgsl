/// Backfill tile_zmin from subtile_zmin
@group(1) @binding(0) var<storage, read> image_heightmap: array<u32>;
@group(1) @binding(1) var<storage, read_write> image_out: array<vec4u>;

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
    let gx = Value(vec4f(1.0, 0.0, 0.0, f32(global_id.x)));
    let gy = Value(vec4f(0.0, 1.0, 0.0, f32(global_id.y)));
    let gz = Value(vec4f(0.0, 0.0, 1.0, f32(z)));

    var ts = array(Value(), Value(), Value(), Value());
    for (var i = 0; i < 4; i++) {
        ts[i] = op_add(
            op_add(
                op_mul(build_imm(config.mat[0][i]), gx),
                op_mul(build_imm(config.mat[1][i]), gy),
            ),
            op_add(
                op_mul(build_imm(config.mat[2][i]), gz),
                build_imm(config.mat[3][i]),
            )
        );
    }
    // Build up input map
    var m = array(Value(), Value(), Value());
    if (config.axes.x < 3) {
        m[config.axes.x] = op_div(ts[0], ts[3]);
    }
    if (config.axes.y < 3) {
        m[config.axes.y] = op_div(ts[1], ts[3]);
    }
    if (config.axes.z < 3) {
        m[config.axes.z] = op_div(ts[2], ts[3]);
    }

    let out = run_tape(0u, m)[0];
    image_out[pixel_index_xy] = vec4u(
        z,
        bitcast<u32>(out.x),
        bitcast<u32>(out.y),
        bitcast<u32>(out.z)
    );
}

struct Value {
    v: vec4f,
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v);
}

fn op_abs(lhs: Value) -> Value {
    if (lhs.v.w < 0.0) {
        return Value(-lhs.v);
    } else {
        return lhs;
    }
}

fn op_recip(lhs: Value) -> Value {
    let v2 = -lhs.v.w * lhs.v.w;
    return Value(vec4f(lhs.v.xyz / v2, 1.0 / lhs.v.w));
}

fn op_sqrt(lhs: Value) -> Value {
    let v = sqrt(lhs.v.w);
    return Value(vec4f(lhs.v.xyz / (2.0 * v), v));
}

fn op_floor(lhs: Value) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, floor(lhs.v.w)));
}

fn op_ceil(lhs: Value) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, ceil(lhs.v.w)));
}

fn op_round(lhs: Value) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, round(lhs.v.w)));
}

fn op_square(lhs: Value) -> Value {
    return op_mul(lhs, lhs);
}

fn op_sin(lhs: Value) -> Value {
    let c = cos(lhs.v.w);
    return Value(vec4f(lhs.v.xyz * c, sin(lhs.v.w)));
}

fn op_cos(lhs: Value) -> Value {
    let s = -sin(lhs.v.w);
    return Value(vec4f(lhs.v.xyz * s, cos(lhs.v.w)));
}

fn op_tan(lhs: Value) -> Value {
    let c = cos(lhs.v.w);
    let c2 = c * c;
    return Value(vec4f(lhs.v.xyz / c, tan(lhs.v.w)));
}

fn op_asin(lhs: Value) -> Value {
    let r = sqrt(1.0 - lhs.v.w * lhs.v.w);
    return Value(vec4f(lhs.v.xyz / r, asin(lhs.v.w)));
}

fn op_acos(lhs: Value) -> Value {
    let r = sqrt(1.0 - lhs.v.w * lhs.v.w);
    return Value(vec4f(-lhs.v.xyz / r, acos(lhs.v.w)));
}

fn op_atan(lhs: Value) -> Value {
    let r = lhs.v.w * lhs.v.w + 1.0;
    return Value(vec4f(lhs.v.xyz / r, atan(lhs.v.w)));
}

fn op_exp(lhs: Value) -> Value {
    let v = exp(lhs.v.w);
    return Value(vec4f(lhs.v.xyz * v, v));
}

fn op_log(lhs: Value) -> Value {
    let v = log(lhs.v.w);
    return Value(vec4f(lhs.v.xyz / v, v));
}

fn op_not(lhs: Value) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, f32(lhs.v.w == 0.0)));
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if (lhs.v.w != lhs.v.w || rhs.v.w != rhs.v.w) {
        return Value(vec4f(0.0, 0.0, 0.0, nan_f32()));
    } else if (lhs.v.w < rhs.v.w) {
        return Value(vec4f(0.0, 0.0, 0.0, -1.0));
    } else if (lhs.v.w > rhs.v.w) {
        return Value(vec4f(0.0, 0.0, 0.0, 1.0));
    } else {
        return Value(vec4f(0.0, 0.0, 0.0, 0.0));
    }
}

fn op_and(lhs: Value, rhs: Value) -> Value {
    if (lhs.v.w != 0.0) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: Value, rhs: Value) -> Value {
    if (lhs.v.w == 0.0) {
        return lhs;
    } else {
        return rhs;
    }
}

fn build_imm(v: f32) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, v));
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

fn op_mod(lhs: Value, rhs: Value) -> Value {
    let e = div_euclid(lhs.v.w, rhs.v.w);
    return Value(vec4f(
        lhs.v.xyz - rhs.v.xyz * e,
        rem_euclid(lhs.v.w, rhs.v.w)
    ));
}

fn op_add(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v + rhs.v);
}

fn op_sub(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v - rhs.v);
}

fn op_min(lhs: Value, rhs: Value) -> Value {
    if (lhs.v.w < rhs.v.w) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_max(lhs: Value, rhs: Value) -> Value {
    if (lhs.v.w > rhs.v.w) {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_mul(lhs: Value, rhs: Value) -> Value {
    return Value(vec4f(
        lhs.v.xyz * rhs.v.w + rhs.v.xyz * lhs.v.w,
        lhs.v.w * rhs.v.w
    ));
}

fn op_div(lhs: Value, rhs: Value) -> Value {
    let d = rhs.v.w * rhs.v.w;
    return Value(vec4f(
        (rhs.v.w * lhs.v.xyz - lhs.v.w * rhs.v.xyz) / d,
        lhs.v.w / rhs.v.w
    ));
}

fn op_atan2(lhs: Value, rhs: Value) -> Value {
    let d = rhs.v.w * rhs.v.w + lhs.v.w * lhs.v.w;
    return Value(vec4f(
        (rhs.v.w * lhs.v.xyz - lhs.v.w * rhs.v.xyz) / d,
        atan2(lhs.v.w, rhs.v.w)
    ));
}
