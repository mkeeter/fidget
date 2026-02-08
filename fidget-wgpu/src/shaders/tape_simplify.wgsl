/// Number of words to allocate for each tape chunk
const CHUNK_SIZE: u32 = 128;

fn simplify_tape(end: u32, tape_len: u32, stack: ptr<function, Stack>) -> u32 {
    // Bail out immediately if there were no choices in the tape
    if !stack.has_choice {
        return 0u;
    }

    var i: u32 = end;
    let chunk_size = min(tape_len * 2, CHUNK_SIZE);
    var chunk_start = alloc(chunk_size);
    var j = chunk_start + chunk_size;
    if j > tape_data.capacity {
        dealloc(chunk_size);
        return 0u;
    }

    var live: array<bool, REG_COUNT>;
    while true {
        i = i - 2;
        j = j - 2; // reserve space

        var op = unpack4xU8(tape_data.data[i]);
        let imm_u = tape_data.data[i + 1];

        if op[0] == OP_JUMP {
            if imm_u == 0xFFFFFFFFu {
                tape_data.data[j] = OP_JUMP;
                tape_data.data[j + 1] = 0xFFFFFFFFu;
                continue;
            } else if imm_u == 0u {
                tape_data.data[j] = OP_JUMP;
                tape_data.data[j + 1] = 0;
                return j;
            } else {
                // Jump to a new tape position
                i = imm_u + 2;
                j += 2; // no allocation happened, so unreserve space
                continue;
            }
        }

        // Allocate a new chunk if needed
        if j == chunk_start {
            chunk_start = alloc(chunk_size);
            let nj = chunk_start + chunk_size - 2;
            if nj >= tape_data.capacity {
                dealloc(chunk_size);
                return 0u;
            }
            tape_data.data[j] = OP_JUMP;
            tape_data.data[j + 1] = nj - 2;
            tape_data.data[nj] = OP_JUMP;
            tape_data.data[nj + 1] = j + 2;
            j = nj - 2;
        }

        if op[0] == OP_OUTPUT {
            // Mark the input register as live
            live[op[1]] = true;
        } else if !live[op[1]] {
            // This is a dead node, so we skip it and pop its choice
            switch op[0] {
                case OP_MIN_REG_IMM,
                OP_MAX_REG_IMM,
                OP_AND_REG_IMM,
                OP_OR_REG_IMM,
                OP_MIN_REG_REG,
                OP_MAX_REG_REG,
                OP_AND_REG_REG,
                OP_OR_REG_REG: {
                    stack_pop(stack);
                }
                default: {
                    // nothing to do here
                }
            }
            j += 2; // no allocation happened, so unreserve space
            continue;
        } else {
            // Mark the output register as unalive
            live[op[1]] = false;
        }

        switch op[0] {
            case OP_OUTPUT: {
                // handled above
            }
            case OP_INPUT, OP_COPY_IMM: {
                // Nothing to do here, we already marked op[1] as unalive
            }

            case OP_COPY_REG, // one input register
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
            OP_NOT_REG,
            OP_ADD_REG_IMM, // one input register, one immediate
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
                live[op[2]] = true;
            }

            // one input register, one immediate, and a choice
            case OP_MIN_REG_IMM,
            OP_MAX_REG_IMM,
            OP_AND_REG_IMM,
            OP_OR_REG_IMM:   {
                switch stack_pop(stack) {
                    case CHOICE_LEFT: {
                        op[0] = OP_COPY_REG; // argument is already in op[2]
                        live[op[2]] = true;
                        // If this is a reassignment, skip it entirely
                        if op[2] == op[1] {
                            j += 2;
                            continue;
                        }
                    }
                    case CHOICE_RIGHT: {
                        op[0] = OP_COPY_IMM; // argument is already in imm_u
                    }
                    default: { // should always be CHOICE_BOTH
                        live[op[2]] = true;
                    }
                }
            }

            // Two input registers
            case OP_ADD_REG_REG,
            OP_MUL_REG_REG,
            OP_DIV_REG_REG,
            OP_SUB_REG_REG,
            OP_COMPARE_REG_REG,
            OP_ATAN_REG_REG,
            OP_MOD_REG_REG: {
                live[op[2]] = true;
                live[op[3]] = true;
            }

            // Two input registers, and a choice
            case OP_MIN_REG_REG,
            OP_MAX_REG_REG,
            OP_AND_REG_REG,
            OP_OR_REG_REG: {
                switch stack_pop(stack) {
                    case CHOICE_LEFT: {
                        op[0] = OP_COPY_REG; // argument is already in op[2]
                        live[op[2]] = true;
                        if op[2] == op[1] { // skip reassignment
                            j += 2;
                            continue;
                        }
                    }
                    case CHOICE_RIGHT: {
                        op[0] = OP_COPY_REG;
                        live[op[3]] = true;
                        op[2] = op[3];
                        if op[2] == op[1] { // skip reassignment
                            j += 2;
                            continue;
                        }
                    }
                    default: { // should always be CHOICE_BOTH
                        live[op[2]] = true;
                        live[op[3]] = true;
                    }
                }
            }

            case OP_JUMP: {
                // handled above
            }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                return 0u;
            }
            default: {
                return 0u;
            }
        }

        // Write the simplified expression back to the new tape
        tape_data.data[j] = pack4xU8(op);
        tape_data.data[j + 1] = imm_u;
    }
    return 0u; // invalid
}

/// Allocates a new chunk, returning the start of the chunk
fn alloc(chunk_size: u32) -> u32 {
    return atomicAdd(&tape_data.offset, chunk_size);
}

/// Undo an allocation
fn dealloc(chunk_size: u32) {
    atomicSub(&tape_data.offset, chunk_size);
}
