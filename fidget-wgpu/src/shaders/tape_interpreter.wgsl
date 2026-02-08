const OP_JUMP: u32 = 0xFF;

// Transform inputs with `config.mat`
fn transformed_inputs(ix: Value, iy: Value, iz: Value) -> array<Value, 3> {
    var ts = array(Value(), Value(), Value(), Value());
    for (var i = 0; i < 4; i++) {
        ts[i] = op_add(
            op_add(
                op_mul(build_imm(config.mat[0][i]), ix),
                op_mul(build_imm(config.mat[1][i]), iy),
            ),
            op_add(
                op_mul(build_imm(config.mat[2][i]), iz),
                build_imm(config.mat[3][i]),
            ),
        );
    }

    // Build up input map
    var m = array(Value(), Value(), Value());
    if config.axes.x < 3 {
        m[config.axes.x] = op_div(ts[0], ts[3]);
    }
    if config.axes.y < 3 {
        m[config.axes.y] = op_div(ts[1], ts[3]);
    }
    if config.axes.z < 3 {
        m[config.axes.z] = op_div(ts[2], ts[3]);
    }
    return m;
}

struct TapeResult {
    value: Value,
    pos: u32,
    count: u32,
}

fn run_tape(start: u32, inputs: array<Value, 3>, stack: ptr<function, Stack>) -> TapeResult {
    var i: u32 = start;
    var count: u32 = 0u;
    var reg: array<Value, REG_COUNT>;

    var out = TapeResult(build_imm(nan_f32()), 0, 0);
    while true {
        count += 1;
        let op = unpack4xU8(tape_data.data[i]);
        let imm_u = tape_data.data[i + 1];
        let imm_v = build_imm(bitcast<f32>(imm_u));
        i = i + 2;
        switch op[0] {
            case OP_OUTPUT: {
                // XXX we're ignoring the output slot here
                out.value = reg[op[1]];
            }
            case OP_INPUT: {
                reg[op[1]] = inputs[imm_u];
            }
            case OP_COPY_REG:    { reg[op[1]] = reg[op[2]]; }
            case OP_COPY_IMM:    { reg[op[1]] = imm_v; }
            case OP_NEG_REG:     { reg[op[1]] = op_neg(reg[op[2]]); }
            case OP_ABS_REG:     { reg[op[1]] = op_abs(reg[op[2]]); }
            case OP_RECIP_REG:   { reg[op[1]] = op_recip(reg[op[2]]); }
            case OP_SQRT_REG:    { reg[op[1]] = op_sqrt(reg[op[2]]); }
            case OP_SQUARE_REG:  { reg[op[1]] = op_square(reg[op[2]]); }
            case OP_FLOOR_REG:   { reg[op[1]] = op_floor(reg[op[2]]); }
            case OP_CEIL_REG:    { reg[op[1]] = op_ceil(reg[op[2]]); }
            case OP_ROUND_REG:   { reg[op[1]] = op_round(reg[op[2]]); }
            case OP_SIN_REG:     { reg[op[1]] = op_sin(reg[op[2]]); }
            case OP_COS_REG:     { reg[op[1]] = op_cos(reg[op[2]]); }
            case OP_TAN_REG:     { reg[op[1]] = op_tan(reg[op[2]]); }
            case OP_ASIN_REG:    { reg[op[1]] = op_asin(reg[op[2]]); }
            case OP_ACOS_REG:    { reg[op[1]] = op_acos(reg[op[2]]); }
            case OP_ATAN_REG:    { reg[op[1]] = op_atan(reg[op[2]]); }
            case OP_EXP_REG:     { reg[op[1]] = op_exp(reg[op[2]]); }
            case OP_LN_REG:      { reg[op[1]] = op_log(reg[op[2]]); }
            case OP_NOT_REG:     { reg[op[1]] = op_not(reg[op[2]]); }
            case OP_ADD_REG_IMM:  { reg[op[1]] = op_add(reg[op[2]], imm_v); }
            case OP_MUL_REG_IMM:  { reg[op[1]] = op_mul(reg[op[2]], imm_v); }
            case OP_DIV_REG_IMM:  { reg[op[1]] = op_div(reg[op[2]], imm_v); }
            case OP_SUB_REG_IMM:  { reg[op[1]] = op_sub(reg[op[2]], imm_v); }
            case OP_MOD_REG_IMM:  { reg[op[1]] = op_mod(reg[op[2]], imm_v); }
            case OP_ATAN_REG_IMM: { reg[op[1]] = op_atan2(reg[op[2]], imm_v); }
            case OP_COMPARE_REG_IMM:  { reg[op[1]] = op_compare(reg[op[2]], imm_v); }

            case OP_DIV_IMM_REG:      { reg[op[1]] = op_div(imm_v, reg[op[2]]); }
            case OP_SUB_IMM_REG:      { reg[op[1]] = op_sub(imm_v, reg[op[2]]); }
            case OP_MOD_IMM_REG:      { reg[op[1]] = op_mod(imm_v, reg[op[2]]); }
            case OP_ATAN_IMM_REG:     { reg[op[1]] = op_atan2(imm_v, reg[op[2]]); }
            case OP_COMPARE_IMM_REG:  { reg[op[1]] = op_compare(imm_v, reg[op[2]]); }

            case OP_MIN_REG_IMM:  { reg[op[1]] = op_min(reg[op[2]], imm_v, stack); }
            case OP_MAX_REG_IMM:  { reg[op[1]] = op_max(reg[op[2]], imm_v, stack); }
            case OP_AND_REG_IMM:  { reg[op[1]] = op_and(reg[op[2]], imm_v, stack); }
            case OP_OR_REG_IMM:   { reg[op[1]] = op_or(reg[op[2]], imm_v, stack); }

            case OP_ADD_REG_REG:      { reg[op[1]] = op_add(reg[op[2]], reg[op[3]]); }
            case OP_MUL_REG_REG:      { reg[op[1]] = op_mul(reg[op[2]], reg[op[3]]); }
            case OP_DIV_REG_REG:      { reg[op[1]] = op_div(reg[op[2]], reg[op[3]]); }
            case OP_SUB_REG_REG:      { reg[op[1]] = op_sub(reg[op[2]], reg[op[3]]); }
            case OP_COMPARE_REG_REG:  { reg[op[1]] = op_compare(reg[op[2]], reg[op[3]]); }
            case OP_ATAN_REG_REG:     { reg[op[1]] = op_atan2(reg[op[2]], reg[op[3]]); }
            case OP_MOD_REG_REG:      { reg[op[1]] = op_mod(reg[op[2]], reg[op[3]]); }

            case OP_MIN_REG_REG:      { reg[op[1]] = op_min(reg[op[2]], reg[op[3]], stack); }
            case OP_MAX_REG_REG:      { reg[op[1]] = op_max(reg[op[2]], reg[op[3]], stack); }
            case OP_AND_REG_REG:      { reg[op[1]] = op_and(reg[op[2]], reg[op[3]], stack); }
            case OP_OR_REG_REG:       { reg[op[1]] = op_or(reg[op[2]], reg[op[3]], stack); }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                return out;
            }

            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu {
                    // end of tape, hope someone wrote `out`
                    out.pos = i;
                    out.count = count;
                    return out;
                } else if imm_u == 0u {
                    // beginning of tape; keep going!
                    continue;
                } else {
                    // Jump to a new tape position
                    i = imm_u;
                }
            }
            default: {
                return out;
            }
        }
    }
    return out;
}
