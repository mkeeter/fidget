/// Compute normals and build a merged image
@group(1) @binding(0) var<storage, read> tape_data: TapeData;
@group(1) @binding(1) var<storage, read> tile64_zmin: array<u32>;
@group(1) @binding(2) var<storage, read> tile16_zmin: array<u32>;
@group(1) @binding(3) var<storage, read> tile4_zmin: array<u32>;
@group(1) @binding(4) var<storage, read> voxels: array<u32>;
@group(1) @binding(5) var<storage, read_write> image_out: array<vec4f>;

@compute @workgroup_size(8, 8)
fn normals_main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    // One compute thread per pixel
    let px = global_id.x;
    let py = global_id.y;
    if px >= config.image_size.x || py >= config.image_size.y {
        return;
    }

    // Pick the highest zmin value available
    let size64 = config.render_size / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;
    let index64 = global_id.x / 64 + global_id.y / 64 * size64.x;
    var out = tile64_zmin[index64];
    let index16 = global_id.x / 16 + global_id.y / 16 * size16.x;
    out = max(out, tile16_zmin[index16]);
    let index4 = global_id.x / 4 + global_id.y / 4 * size4.x;
    out = max(out, tile4_zmin[index4]);
    out = max(out, voxels[global_id.x + global_id.y * config.render_size[0]]);

    let z = out >> 20;
    let tape_index = out & ((1 << 20) - 1);

    // If this pixel hasn't yet been written in the heightmap, then return
    if z == 0 {
        return;
    }
    image_out[px + py * config.image_size.x] = vec4f(4.0, 5.0, 6.0, 7.0);

    let pixel_index_xy = px + py * config.image_size.x;

    // Store gradients with dx, dy, dz in xyz and value in w
    let gx = Value(vec4f(1.0, 0.0, 0.0, f32(px)));
    let gy = Value(vec4f(0.0, 1.0, 0.0, f32(py)));
    let gz = Value(vec4f(0.0, 0.0, 1.0, f32(z)));

    // Compute input values
    let m = transformed_inputs(gx, gy, gz);

    var stack = Stack(); // dummy value
    let result = run_tape(tape_index, m, &stack);
    image_out[pixel_index_xy] = vec4f(
        f32(z),
        result.value.v.x,
        result.value.v.y,
        result.value.v.z
    );
}

struct Value {
    v: vec4f,
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v);
}

fn op_abs(lhs: Value) -> Value {
    if lhs.v.w < 0.0 {
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
    return Value(vec4f(lhs.v.xyz / c2, tan(lhs.v.w)));
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
    return Value(vec4f(lhs.v.xyz / lhs.v.w, v));
}

fn op_not(lhs: Value) -> Value {
    return Value(vec4f(0.0, 0.0, 0.0, f32(lhs.v.w == 0.0)));
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if lhs.v.w != lhs.v.w || rhs.v.w != rhs.v.w {
        return Value(vec4f(0.0, 0.0, 0.0, nan_f32()));
    } else if lhs.v.w < rhs.v.w {
        return Value(vec4f(0.0, 0.0, 0.0, -1.0));
    } else if lhs.v.w > rhs.v.w {
        return Value(vec4f(0.0, 0.0, 0.0, 1.0));
    } else {
        return Value(vec4f(0.0, 0.0, 0.0, 0.0));
    }
}

fn op_and(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v.w == 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v.w != 0.0 {
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
    if r < 0.0 {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn div_euclid(lhs: f32, rhs: f32) -> f32 {
    let q = trunc(lhs / rhs);
    if lhs % rhs < 0.0 {
        if rhs > 0.0 {
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

fn op_min(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v.w < rhs.v.w {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_max(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v.w > rhs.v.w {
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
