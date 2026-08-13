const OP_JUMP: u32 = 0xFF;

struct TapeResult {
    value: Value,
    pos: u32,
    count: u32,
}

fn run_tape(start: u32, xyz: array<Value, 3>, stack: ptr<function, Stack>) -> TapeResult {
    var i: u32 = start;
    var count: u32 = 0u;
    var reg: array<Value, REG_COUNT>;

    var lhs = Value();
    var rhs = Value();
    var out = TapeResult(build_imm(nan_f32()), 0, 0);
    while true {
        count += 1;
        let word = config.tape_data[i];
        let op = unpack4xU8(word.op);
        let rhs_i = op[3];
        let lhs_i = op[2];
        let imm_u = word.imm;
        let imm_v = build_imm(bitcast<f32>(imm_u));
        if lhs_i == 255 {
            lhs = imm_v;
        } else {
            lhs = reg[lhs_i];
        }
        if rhs_i == 255 {
            rhs = imm_v;
        } else {
            rhs = reg[rhs_i];
        }
        var tmp = build_imm(0.0);
        i = i + 1;
        switch op[0] {
            case OP_OUTPUT: {
                // XXX we're ignoring the output slot here
                out.value = reg[op[1]];
                continue;
            }
            case OP_INPUT: {
                if imm_u == config.axes.x {
                    tmp = xyz[0u];
                } else if imm_u == config.axes.y {
                    tmp = xyz[1u];
                } else if imm_u == config.axes.z {
                    tmp = xyz[2u];
                } else {
                    tmp = build_imm(var_values[imm_u]);
                }
            }
            case OP_COPY:    { tmp = lhs; }
            case OP_NEG:     { tmp = op_neg(lhs); }
            case OP_ABS:     { tmp = op_abs(lhs); }
            case OP_RECIP:   { tmp = op_recip(lhs); }
            case OP_SQRT:    { tmp = op_sqrt(lhs); }
            case OP_SQUARE:  { tmp = op_square(lhs); }
            case OP_FLOOR:   { tmp = op_floor(lhs); }
            case OP_CEIL:    { tmp = op_ceil(lhs); }
            case OP_ROUND:   { tmp = op_round(lhs); }
            case OP_SIN:     { tmp = op_sin(lhs); }
            case OP_COS:     { tmp = op_cos(lhs); }
            case OP_TAN:     { tmp = op_tan(lhs); }
            case OP_ASIN:    { tmp = op_asin(lhs); }
            case OP_ACOS:    { tmp = op_acos(lhs); }
            case OP_ATAN:    { tmp = op_atan(lhs); }
            case OP_EXP:     { tmp = op_exp(lhs); }
            case OP_LN:      { tmp = op_log(lhs); }
            case OP_NOT:     { tmp = op_not(lhs); }
            case OP_RAND:    { tmp = op_rand(lhs); }

            case OP_ADD:     { tmp = op_add(lhs, rhs); }
            case OP_MUL:     { tmp = op_mul(lhs, rhs); }
            case OP_DIV:     { tmp = op_div(lhs, rhs); }
            case OP_SUB:     { tmp = op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = op_compare(lhs, rhs); }
            case OP_ATAN2:   { tmp = op_atan2(lhs, rhs); }
            case OP_MOD:     { tmp = op_mod(lhs, rhs); }
            case OP_MIX:     { tmp = op_mix(lhs, rhs); }

            case OP_MIN:     { tmp = op_min(lhs, rhs, stack); }
            case OP_MAX:     { tmp = op_max(lhs, rhs, stack); }
            case OP_AND:     { tmp = op_and(lhs, rhs, stack); }
            case OP_OR:      { tmp = op_or(lhs, rhs, stack); }

            case OP_MEM: {
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
                    continue;
                }
            }
            default: {
                return out;
            }
        }
        reg[op[1]] = tmp;
    }
    return out;
}
