/// WGSL fragment for running an interpreter on 4x floats
//
// `OP_*` constants are generated at runtime based on bytecode format, so this
// shader cannot be compiled as-is.

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7FC00000);
}

fn compare_f32(lhs: f32, rhs: f32) -> f32 {
    if (lhs < rhs) {
        return -1.0;
    } else if (lhs > rhs) {
        return 1.0;
    } else if (lhs == rhs) {
        return 0.0;
    } else {
        return nan_f32();
    }
}

fn compare_4f(lhs: vec4f, rhs: vec4f) -> vec4f {
    var out = vec4f(0.0);
    for (var i=0; i < 4; i += 1) {
        out[i] = compare_f32(lhs[i], rhs[i]);
    }
    return out;
}

fn and_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs == 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn and_4f(lhs: vec4f, rhs: vec4f) -> vec4f {
    var out = vec4f(0.0);
    for (var i=0; i < 4; i += 1) {
        out[i] = and_f32(lhs[i], rhs[i]);
    }
    return out;
}

fn or_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs != 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn or_4f(lhs: vec4f, rhs: vec4f) -> vec4f {
    var out = vec4f(0.0);
    for (var i=0; i < 4; i += 1) {
        out[i] = or_f32(lhs[i], rhs[i]);
    }
    return out;
}

fn not_f32(lhs: f32) -> f32 {
    return f32(lhs != 0.0);
}

fn not_4f(lhs: vec4f) -> vec4f {
    var out = vec4f(0.0);
    for (var i=0; i < 4; i += 1) {
        out[i] = not_f32(lhs[i]);
    }
    return out;
}

fn read_imm_4f(i: ptr<function, u32>) -> vec4f {
    let imm = bitcastf(tape[*i]);
    *i = *i + 1;
    return vec4f(imm);
}

fn run_tape(start: u32, inputs: mat4x4f) -> vec4f {
    var i: u32 = start;
    var reg: array<vec4f, 256>;
    while (true) {
        let op = unpack4xU8(tape[i]);
        i = i + 1;
        switch op[0] {
            case OP_OUTPUT: {
                // XXX we're not actually writing to an output slot here
                let imm = tape[i];
                i = i + 1;
                return reg[op[1]];
            }
            case OP_INPUT: {
                let imm = tape[i];
                i = i + 1;
                reg[op[1]] = inputs[imm];
            }
            case OP_COPY_REG:    { reg[op[1]] = reg[op[2]]; }
            case OP_COPY_IMM:    { reg[op[1]] = read_imm_4f(&i); }
            case OP_NEG_REG:     { reg[op[1]] = -reg[op[2]]; }
            case OP_ABS_REG:     { reg[op[1]] = abs(reg[op[2]]); }
            case OP_RECIP_REG:   { reg[op[1]] = 1.0 / reg[op[2]]; }
            case OP_SQRT_REG:    { reg[op[1]] = sqrt(reg[op[2]]); }
            case OP_SQUARE_REG: {
                let v = reg[op[2]];
                reg[op[1]] = v * v;
            }
            case OP_FLOOR_REG:   { reg[op[1]] = floor(reg[op[2]]); }
            case OP_CEIL_REG:    { reg[op[1]] = ceil(reg[op[2]]); }
            case OP_ROUND_REG:   { reg[op[1]] = round(reg[op[2]]); }
            case OP_SIN_REG:     { reg[op[1]] = sin(reg[op[2]]); }
            case OP_COS_REG:     { reg[op[1]] = cos(reg[op[2]]); }
            case OP_TAN_REG:     { reg[op[1]] = tan(reg[op[2]]); }
            case OP_ASIN_REG:    { reg[op[1]] = asin(reg[op[2]]); }
            case OP_ACOS_REG:    { reg[op[1]] = acos(reg[op[2]]); }
            case OP_ATAN_REG:    { reg[op[1]] = atan(reg[op[2]]); }
            case OP_EXP_REG:     { reg[op[1]] = exp(reg[op[2]]); }
            case OP_LN_REG:      { reg[op[1]] = log(reg[op[2]]); }
            case OP_NOT_REG:     { reg[op[1]] = not_4f(reg[op[2]]); }
            case OP_ADD_REG_IMM:  { reg[op[1]] = reg[op[2]] + read_imm_4f(&i); }
            case OP_MUL_REG_IMM:  { reg[op[1]] = reg[op[2]] * read_imm_4f(&i); }
            case OP_DIV_REG_IMM:  { reg[op[1]] = reg[op[2]] / read_imm_4f(&i); }
            case OP_SUB_REG_IMM:  { reg[op[1]] = reg[op[2]] - read_imm_4f(&i); }
            case OP_MOD_REG_IMM:  { reg[op[1]] = reg[op[2]] % read_imm_4f(&i); }
            case OP_ATAN_REG_IMM: { reg[op[1]] = atan2(reg[op[2]], read_imm_4f(&i)); }
            case OP_COMPARE_REG_IMM:  { reg[op[1]] = compare_4f(reg[op[2]], read_imm_4f(&i)); }

            case OP_DIV_IMM_REG:      { reg[op[1]] = read_imm_4f(&i) / reg[op[2]]; }
            case OP_SUB_IMM_REG:      { reg[op[1]] = read_imm_4f(&i) - reg[op[2]]; }
            case OP_MOD_IMM_REG:      { reg[op[1]] = read_imm_4f(&i) % reg[op[2]]; }
            case OP_ATAN_IMM_REG:     { reg[op[1]] = atan2(read_imm_4f(&i), reg[op[2]]); }
            case OP_COMPARE_IMM_REG:  { reg[op[1]] = compare_4f(read_imm_4f(&i), reg[op[2]]); }

            case OP_MIN_REG_IMM:  { reg[op[1]] = min(reg[op[2]], read_imm_4f(&i)); }
            case OP_MAX_REG_IMM:  { reg[op[1]] = max(reg[op[2]], read_imm_4f(&i)); }
            case OP_AND_REG_IMM:  { reg[op[1]] = and_4f(reg[op[2]], read_imm_4f(&i)); }
            case OP_OR_REG_IMM:   { reg[op[1]] = or_4f(reg[op[2]], read_imm_4f(&i)); }

            case OP_ADD_REG_REG:      { reg[op[1]] = reg[op[2]] + reg[op[3]]; }
            case OP_MUL_REG_REG:      { reg[op[1]] = reg[op[2]] * reg[op[3]]; }
            case OP_DIV_REG_REG:      { reg[op[1]] = reg[op[2]] / reg[op[3]]; }
            case OP_SUB_REG_REG:      { reg[op[1]] = reg[op[2]] - reg[op[3]]; }
            case OP_COMPARE_REG_REG:  { reg[op[1]] = reg[op[2]] - reg[op[3]]; }
            case OP_ATAN_REG_REG:      { reg[op[1]] = atan2(reg[op[2]], reg[op[3]]); }
            case OP_MOD_REG_REG:      { reg[op[1]] = reg[op[2]] % reg[op[3]]; }

            case OP_MIN_REG_REG:      { reg[op[1]] = min(reg[op[2]], reg[op[3]]); }
            case OP_MAX_REG_REG:      { reg[op[1]] = max(reg[op[2]], reg[op[3]]); }
            case OP_AND_REG_REG:      { reg[op[1]] = and_4f(reg[op[2]], reg[op[3]]); }
            case OP_OR_REG_REG:       { reg[op[1]] = or_4f(reg[op[2]], reg[op[3]]); }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                break;
            }
            default: {
                break;
            }
        }
    }
    return vec4f(nan_f32()); // unknown opcode
}
