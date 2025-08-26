/// Number of words to allocate for each tape chunk
const CHUNK_SIZE: u32 = 128;

/// Allocates a new chunk, returning a past-the-end pointer
fn alloc() -> u32 {
    return atomicAdd(&config.tape_data_offset, CHUNK_SIZE) + CHUNK_SIZE;
}

fn simplify_tape(end: u32, stack: ptr<function, Stack>) -> u32 {
    var i: u32 = end;
    var live: array<bool, 256>;
    var j = alloc();
    if j > config.tape_data_capacity {
        return 0u;
    }

    while (true) {
        i = i - 2;
        j = j - 2; // reserve space

        let new_op = vec2u(config.tape_data[i], config.tape_data[i + 1]);
        let op = unpack4xU8(new_op.x);

        if op[0] == OP_JUMP {
            let imm_u = new_op.y;
            if imm_u == 0xFFFFFFFFu {
                config.tape_data[j] = OP_JUMP;
                config.tape_data[j + 1] = 0xFFFFFFFFu;
                continue;
            } else if imm_u == 0u {
                config.tape_data[j] = OP_JUMP;
                config.tape_data[j + 1] = 0;
                return j;
            } else {
                // Jump to a new tape position
                i = imm_u;
                j += 2; // no allocation happened, so unreserve space
                continue;
            }
        }

        // Allocate a new chunk if needed
        if j == 0 {
            let nj = alloc() - 2;
            if nj >= config.tape_data_capacity {
                return 0u;
            }
            config.tape_data[j] = OP_JUMP;
            config.tape_data[j + 1] = nj - 2;
            config.tape_data[nj] = OP_JUMP;
            config.tape_data[nj + 1] = j + 2;
            j = nj - 2;
        }

        if op[0] == OP_OUTPUT {
            live[op[1]] = true;
        }

        if !live[op[1]] {
            // TODO pop stack
            continue;
        }

        switch op[0] {
            case OP_OUTPUT: {
                // handled above
            }
            case OP_INPUT, OP_COPY_IMM: {
                live[op[1]] = false;
            }
            case OP_COPY_REG,
            OP_NEG_REG,
            OP_ABS_REG,
            OP_RECIP_REG,
            OP_SQRT_REG,
            OP_SQUARE_REG,
            OP_FLOOR_REG,
            OP_CEIL_REG,
            OP_ROUND_REG,
            OP_SIN_REG,
            OP_COS_REG,
            OP_TAN_REG,
            OP_ASIN_REG,
            OP_ACOS_REG,
            OP_ATAN_REG,
            OP_EXP_REG,
            OP_LN_REG,
            OP_NOT_REG: {
                live[op[1]] = false;
                live[op[2]] = true;
            }

            case OP_ADD_REG_IMM,
            OP_MUL_REG_IMM,
            OP_DIV_REG_IMM,
            OP_SUB_REG_IMM,
            OP_MOD_REG_IMM,
            OP_ATAN_REG_IMM,
            OP_COMPARE_REG_IMM,
            OP_DIV_IMM_REG,
            OP_SUB_IMM_REG,
            OP_MOD_IMM_REG,
            OP_ATAN_IMM_REG,
            OP_COMPARE_IMM_REG: {
                live[op[1]] = false;
                live[op[2]] = true;
            }

            case OP_MIN_REG_IMM,
            OP_MAX_REG_IMM,
            OP_AND_REG_IMM,
            OP_OR_REG_IMM:   {
                // TODO handle choices here
                live[op[1]] = false;
                live[op[2]] = true;
            }

            case OP_ADD_REG_REG,
            OP_MUL_REG_REG,
            OP_DIV_REG_REG,
            OP_SUB_REG_REG,
            OP_COMPARE_REG_REG,
            OP_ATAN_REG_REG,
            OP_MOD_REG_REG: {
                live[op[1]] = false;
                live[op[2]] = true;
                live[op[3]] = true;
            }

            case OP_MIN_REG_REG,
            OP_MAX_REG_REG,
            OP_AND_REG_REG,
            OP_OR_REG_REG: {
                // TODO handle choices here
                live[op[1]] = false;
                live[op[2]] = true;
                live[op[3]] = true;
            }

            case OP_JUMP: {
                // handled above
            }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                break;
            }
            default: {
                break;
            }
        }

        // Write the simplified expression back to the new tape
        config.tape_data[j] = new_op.x;
        config.tape_data[j + 1] = new_op.y;
    }
    return 0u; // invalid
}
