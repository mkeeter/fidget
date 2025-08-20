fn read_imm(i: ptr<function, u32>) -> Value {
    let out = bitcast<f32>(config.tape_data[*i]);
    *i = *i + 1;
    return build_imm(out);
}

fn run_tape(start: u32, inputs: array<Value, 3>, stack: ptr<function, Stack>) -> Value {
    var i: u32 = start;
    var reg: array<Value, 256>;
    while (true) {
        let op = unpack4xU8(config.tape_data[i]);
        i = i + 1;
        switch op[0] {
            case OP_OUTPUT: {
                // XXX we're not actually writing to an output slot here
                let imm = config.tape_data[i];
                i = i + 1;
                return reg[op[1]];
            }
            case OP_INPUT: {
                let imm = config.tape_data[i];
                i = i + 1;
                reg[op[1]] = inputs[imm];
            }
            case OP_COPY_REG:    { reg[op[1]] = reg[op[2]]; }
            case OP_COPY_IMM:    { reg[op[1]] = read_imm(&i); }
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
            case OP_ADD_REG_IMM:  { reg[op[1]] = op_add(reg[op[2]], read_imm(&i)); }
            case OP_MUL_REG_IMM:  { reg[op[1]] = op_mul(reg[op[2]], read_imm(&i)); }
            case OP_DIV_REG_IMM:  { reg[op[1]] = op_div(reg[op[2]], read_imm(&i)); }
            case OP_SUB_REG_IMM:  { reg[op[1]] = op_sub(reg[op[2]], read_imm(&i)); }
            case OP_MOD_REG_IMM:  { reg[op[1]] = op_mod(reg[op[2]], read_imm(&i)); }
            case OP_ATAN_REG_IMM: { reg[op[1]] = op_atan2(reg[op[2]], read_imm(&i)); }
            case OP_COMPARE_REG_IMM:  { reg[op[1]] = op_compare(reg[op[2]], read_imm(&i)); }

            case OP_DIV_IMM_REG:      { reg[op[1]] = op_div(read_imm(&i), reg[op[2]]); }
            case OP_SUB_IMM_REG:      { reg[op[1]] = op_sub(read_imm(&i), reg[op[2]]); }
            case OP_MOD_IMM_REG:      { reg[op[1]] = op_mod(read_imm(&i), reg[op[2]]); }
            case OP_ATAN_IMM_REG:     { reg[op[1]] = op_atan2(read_imm(&i), reg[op[2]]); }
            case OP_COMPARE_IMM_REG:  { reg[op[1]] = op_compare(read_imm(&i), reg[op[2]]); }

            case OP_MIN_REG_IMM:  { reg[op[1]] = op_min(reg[op[2]], read_imm(&i), stack); }
            case OP_MAX_REG_IMM:  { reg[op[1]] = op_max(reg[op[2]], read_imm(&i), stack); }
            case OP_AND_REG_IMM:  { reg[op[1]] = op_and(reg[op[2]], read_imm(&i)); }
            case OP_OR_REG_IMM:   { reg[op[1]] = op_or(reg[op[2]], read_imm(&i)); }

            case OP_ADD_REG_REG:      { reg[op[1]] = op_add(reg[op[2]], reg[op[3]]); }
            case OP_MUL_REG_REG:      { reg[op[1]] = op_mul(reg[op[2]], reg[op[3]]); }
            case OP_DIV_REG_REG:      { reg[op[1]] = op_div(reg[op[2]], reg[op[3]]); }
            case OP_SUB_REG_REG:      { reg[op[1]] = op_sub(reg[op[2]], reg[op[3]]); }
            case OP_COMPARE_REG_REG:  { reg[op[1]] = op_compare(reg[op[2]], reg[op[3]]); }
            case OP_ATAN_REG_REG:     { reg[op[1]] = op_atan2(reg[op[2]], reg[op[3]]); }
            case OP_MOD_REG_REG:      { reg[op[1]] = op_mod(reg[op[2]], reg[op[3]]); }

            case OP_MIN_REG_REG:      { reg[op[1]] = op_min(reg[op[2]], reg[op[3]], stack); }
            case OP_MAX_REG_REG:      { reg[op[1]] = op_max(reg[op[2]], reg[op[3]], stack); }
            case OP_AND_REG_REG:      { reg[op[1]] = op_and(reg[op[2]], reg[op[3]]); }
            case OP_OR_REG_REG:       { reg[op[1]] = op_or(reg[op[2]], reg[op[3]]); }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                return build_imm(nan_f32());
            }
            default: {
                return build_imm(nan_f32());
            }
        }
    }
    return build_imm(nan_f32()); // unknown opcode
}

