struct Value {
    v: f32,
}

fn build_imm(imm: f32) -> Value {
    return Value(imm);
}

fn op_abs(lhs: Value) -> Value {
    return Value(abs(lhs.v));
}

fn op_acos(lhs: Value) -> Value {
    return Value(acos(lhs.v));
}

fn op_cos(lhs: Value) -> Value {
    return Value(cos(lhs.v));
}

fn op_asin(lhs: Value) -> Value {
    return Value(asin(lhs.v));
}

fn op_atan(lhs: Value) -> Value {
    return Value(atan(lhs.v));
}

fn op_ceil(lhs: Value) -> Value {
    return Value(ceil(lhs.v));
}

fn op_floor(lhs: Value) -> Value {
    return Value(floor(lhs.v));
}

fn op_log(lhs: Value) -> Value {
    return Value(log(lhs.v));
}

fn op_recip(lhs: Value) -> Value {
    return Value(1.0 / lhs.v);
}

fn op_round(lhs: Value) -> Value {
    return Value(round(lhs.v));
}

fn op_sin(lhs: Value) -> Value {
    return Value(sin(lhs.v));
}

fn op_tan(lhs: Value) -> Value {
    return Value(tan(lhs.v));
}

fn op_exp(lhs: Value) -> Value {
    return Value(exp(lhs.v));
}

fn op_add(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v + rhs.v);
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v);
}

fn op_sub(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v - rhs.v);
}

fn op_mul(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v * rhs.v);
}

fn op_div(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v / rhs.v);
}

fn op_atan2(lhs: Value, rhs: Value) -> Value {
    return Value(atan2(lhs.v, rhs.v));
}

fn op_min(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    return Value(min(lhs.v, rhs.v));
}

fn op_max(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    return Value(max(lhs.v, rhs.v));
}

fn op_square(lhs: Value) -> Value {
    return Value(lhs.v * lhs.v);
}

fn op_sqrt(lhs: Value) -> Value {
    return Value(sqrt(lhs.v));
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if lhs.v < rhs.v {
        return Value(-1.0);
    } else if lhs.v > rhs.v {
        return Value(1.0);
    } else if lhs.v == rhs.v {
        return Value(0.0);
    } else {
        return Value(nan_f32());
    }
}

fn op_and(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v == 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v != 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_not(lhs: Value) -> Value {
    return Value(f32(lhs.v == 0.0));
}

fn op_mod(lhs: Value, rhs: Value) -> Value {
    return Value(rem_euclid(lhs.v, rhs.v));
}
