@group(1) @binding(0) var<storage, read> image_heightmap: array<u32>;
@group(1) @binding(1) var<storage, read_write> image_out: array<vec4f>;

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
    let pixel_index_xy = px + py * config.image_size.x;

    // If we've already written this pixel, then return; because evaluation
    // happens in Z order, it will necessarily supersede the current pixel
    if image_out[pixel_index_xy][0] != 0 {
        return;
    }

    // If this pixel hasn't yet been written in the heightmap, then return
    let z = image_heightmap[pixel_index_xy];
    if z == 0u {
        return;
    }

    // Store gradients with dx, dy, dz in xyz and value in w
    let gx = Value(vec4f(1.0, 0.0, 0.0, f32(px)));
    let gy = Value(vec4f(0.0, 1.0, 0.0, f32(py)));
    let gz = Value(vec4f(0.0, 0.0, 1.0, f32(z)));

    // Compute input values
    let m = transformed_inputs(gx, gy, gz);

    let tape_start = get_tape_start(vec3u(px, py, z));
    var stack = Stack(); // dummy value
    let out = run_tape(tape_start, m, &stack);
    image_out[pixel_index_xy] = vec4f(
        f32(z),
        out.value.v.x,
        out.value.v.y,
        out.value.v.z
    );
}

/// For a given voxel position, return the tape start index
///
/// This is the highest-resolution tape index that is valid for the given
/// position, e.g. preferring tapes specialized to 4x4x4 regions, then
/// 16x16x16, then 64x64x64.
fn get_tape_start(corner_pos: vec3u) -> u32 {
    // The 4^3 and 16^3 tiles are reused between strata, so we have to check Z
    let index4 = get_tape_offset_for_level(corner_pos, 4u);
    // that the Z position is valid for the current strata.
    let t4 = tile_tape[index4];
    if t4 != 0 && t4 >> TILE_TAPE_STRATA_SHIFT == corner_pos.z / 4 {
        return t4 & TILE_TAPE_MASK;
    }

    let index16 = get_tape_offset_for_level(corner_pos, 16u);
    let t16 = tile_tape[index16];
    if t16 != 0 && t16 >> TILE_TAPE_STRATA_SHIFT == corner_pos.z / 16 {
        return t16 & TILE_TAPE_MASK;
    }

    let index64 = get_tape_offset_for_level(corner_pos, 64u);
    let t64 = tile_tape[index64];
    if t64 != 0 {
        return t64;
    } else {
        return 0u;
    }
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
