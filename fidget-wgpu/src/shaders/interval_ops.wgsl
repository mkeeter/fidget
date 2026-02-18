
struct Value {
    v: vec2f,
}

fn is_nan(v: f32) -> bool {
    let u = bitcast<u32>(v);
    let exponent = (u >> 23u) & 0xFFu;
    let mantissa = u & 0x7FFFFFu;
    return (exponent == 0xFFu) && (mantissa != 0u);
}

fn has_nan(i: Value) -> bool {
    return is_nan(i.v.x) || is_nan(i.v.y);
}

fn nan_i() -> Value {
    return Value(vec2f(nan_f32()));
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v.yx);
}

fn op_abs(lhs: Value) -> Value {
    if lhs.v[0] < 0.0 {
        if lhs.v[1] > 0.0 {
            return Value(vec2f(0.0, max(lhs.v[1], -lhs.v[0])));
        } else {
            return Value(vec2f(-lhs.v[1], -lhs.v[0]));
        }
    } else {
        return lhs;
    }
}

fn op_recip(lhs: Value) -> Value {
    if lhs.v[0] > 0.0 || lhs.v[1] < 0.0 {
        return Value(vec2f(1.0 / lhs.v[1], 1.0 / lhs.v[0]));
    } else {
        return nan_i();
    }
}

fn op_sqrt(lhs: Value) -> Value {
    if lhs.v[0] >= 0.0 {
        return Value(sqrt(lhs.v));
    } else {
        return nan_i();
    }
}

fn op_floor(lhs: Value) -> Value {
    return Value(floor(lhs.v));
}

fn op_ceil(lhs: Value) -> Value {
    return Value(ceil(lhs.v));
}

fn op_round(lhs: Value) -> Value {
    return Value(round(lhs.v));
}

fn op_square(lhs: Value) -> Value {
    if lhs.v[1] < 0.0 {
        return Value(vec2f(lhs.v[1] * lhs.v[1], lhs.v[0] * lhs.v[0]));
    } else if lhs.v[0] > 0.0 {
        return Value(vec2f(lhs.v[0] * lhs.v[0], lhs.v[1] * lhs.v[1]));
    } else if has_nan(lhs) {
        return nan_i();
    } else {
        let v = max(abs(lhs.v[0]), abs(lhs.v[1]));
        return Value(vec2f(0.0, v * v));
    }
}

fn op_sin(lhs: Value) -> Value {
    if has_nan(lhs) {
        return nan_i();
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_cos(lhs: Value) -> Value {
    if has_nan(lhs) {
        return nan_i();
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_tan(lhs: Value) -> Value {
    let size = lhs.v[1] - lhs.v[0];
    if size >= 3.14159265f {
        return nan_i();
    } else {
        let lower = tan(lhs.v[0]);
        let upper = tan(lhs.v[1]);
        if upper >= lower {
            return Value(vec2f(lower, upper));
        } else {
            return nan_i();
        }
    }
}

fn op_asin(lhs: Value) -> Value {
    if lhs.v[0] < -1.0 || lhs.v[1] > 1.0 {
        return nan_i();
    } else {
        return Value(asin(lhs.v));
    }
}

fn op_acos(lhs: Value) -> Value {
    if lhs.v[0] < -1.0 || lhs.v[1] > 1.0 {
        return nan_i();
    } else {
        return Value(acos(lhs.v).yx);
    }
}

fn op_atan(lhs: Value) -> Value {
    return Value(atan(lhs.v));
}

fn op_exp(lhs: Value) -> Value {
    return Value(exp(lhs.v));
}

fn op_log(lhs: Value) -> Value {
    if lhs.v[0] < 0.0 {
        return nan_i();
    } else {
        return Value(log(lhs.v));
    }
}

fn op_not(lhs: Value) -> Value {
    if !contains_i(lhs, 0.0) && !has_nan(lhs) {
        return Value(vec2f(0.0));
    } else if lhs.v[0] == 0.0 && lhs.v[1] == 0.0 {
        return Value(vec2f(1.0));
    } else {
        return Value(vec2f(0.0, 1.0));
    }
}

fn contains_i(i: Value, v: f32) -> bool {
    return (i.v[0] <= v && v <= i.v[1]);
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        return nan_i();
    } else if lhs.v[1] < rhs.v[0] {
        return Value(vec2f(-1.0));
    } else if lhs.v[0] > rhs.v[1] {
        return Value(vec2f(1.0));
    } else if lhs.v[0] == lhs.v[1] && rhs.v[0] == rhs.v[1] && lhs.v[0] == rhs.v[0] {
        return Value(vec2f(0.0));
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_and(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        stack_push(stack, CHOICE_BOTH);
        return nan_i();
    } else if lhs.v[0] == 0.0 && lhs.v[1] == 0.0 {
        stack_push(stack, CHOICE_LEFT);
        return Value(vec2f(0.0));
    } else if !contains_i(lhs, 0.0) {
        stack_push(stack, CHOICE_RIGHT);
        return rhs;
    } else {
        stack_push(stack, CHOICE_BOTH);
        return Value(vec2f(min(rhs.v[0], 0.0), max(rhs.v[1], 0.0)));
    }
}

fn op_or(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        stack_push(stack, CHOICE_BOTH);
        return nan_i();
    } else if !contains_i(lhs, 0.0) {
        stack_push(stack, CHOICE_LEFT);
        return lhs;
    } else if lhs.v[0] == 0.0 && lhs.v[1] == 0.0 {
        stack_push(stack, CHOICE_RIGHT);
        return rhs;
    } else {
        stack_push(stack, CHOICE_BOTH);
        return Value(vec2f(min(lhs.v[0], rhs.v[0]), max(lhs.v[1], rhs.v[1])));
    }
}

fn build_imm(v: f32) -> Value {
    return Value(vec2f(v));
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if r < 0.0 {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn op_mod(lhs: Value, rhs: Value) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        return nan_i();
    } else if rhs.v[0] == rhs.v[1] && rhs.v[0] > 0.0 {
        let a = lhs.v[0] / rhs.v[0];
        let b = lhs.v[1] / rhs.v[0];
        if a != floor(a) && floor(a) == floor(b) {
            return Value(vec2f(
                rem_euclid(lhs.v[0], rhs.v[0]),
                rem_euclid(lhs.v[1], rhs.v[0]),
            ));
        } else {
            return Value(vec2f(0.0, abs(rhs.v[0])));
        }
    } else {
        return Value(vec2f(0.0, max(abs(rhs.v[0]), abs(rhs.v[1]))));
    }
}

fn op_add(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v + rhs.v);
}

fn op_sub(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v - rhs.v.yx);
}

fn op_min(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        stack_push(stack, CHOICE_BOTH);
        return nan_i();
    } else if lhs.v[1] < rhs.v[0] {
        stack_push(stack, CHOICE_LEFT);
        return lhs;
    } else if rhs.v[1] < lhs.v[0] {
        stack_push(stack, CHOICE_RIGHT);
        return rhs;
    } else {
        stack_push(stack, CHOICE_BOTH);
        return Value(min(lhs.v, rhs.v));
    }
}

fn op_max(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        stack_push(stack, CHOICE_BOTH);
        return nan_i();
    } else if lhs.v[0] > rhs.v[1] {
        stack_push(stack, CHOICE_LEFT);
        return lhs;
    } else if rhs.v[0] > lhs.v[1] {
        stack_push(stack, CHOICE_RIGHT);
        return rhs;
    } else {
        stack_push(stack, CHOICE_BOTH);
        return Value(max(lhs.v, rhs.v));
    }
}

fn op_mul(lhs: Value, rhs: Value) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        return nan_i();
    }
    let ab = lhs.v * rhs.v;
    let cd = lhs.v.yx * rhs.v;
    return Value(vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    ));
}

fn op_div(lhs: Value, rhs: Value) -> Value {
    if has_nan(lhs) || contains_i(rhs, 0.0) {
        return nan_i();
    }
    let ab = lhs.v / rhs.v;
    let cd = lhs.v.yx / rhs.v;
    return Value(vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    ));
}

fn op_atan2(lhs: Value, rhs: Value) -> Value {
    if has_nan(lhs) || has_nan(rhs) {
        return nan_i();
    }
    return Value(vec2f(-3.141592654, 3.141592654));
}

fn interval_inputs(tile_corner: vec3u, tile_size: u32) -> array<Value, 3> {
    // Tile corner position, in voxels
    let corner_pos = tile_corner * tile_size;

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + tile_size));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + tile_size));
    let iz = vec2f(f32(corner_pos.z), f32(corner_pos.z + tile_size));

    return transformed_inputs(Value(ix), Value(iy), Value(iz));
}
