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
                case OP_MIN,
                OP_MAX,
                OP_AND,
                OP_OR: {
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
            case OP_INPUT {
                // Nothing to do here, we already marked op[1] as unalive
            }

            // One input register (or immediate)
            case OP_COPY,
            OP_NEG,
            OP_ABS,
            OP_RECIP,
            OP_SQRT,
            OP_SQUARE,
            OP_FLOOR,
            OP_CEIL,
            OP_ROUND,
            OP_SIN,
            OP_COS,
            OP_TAN,
            OP_ASIN,
            OP_ACOS,
            OP_ATAN,
            OP_EXP,
            OP_LN,
            OP_NOT: {
                if op[2] != 255 {
                    live[op[2]] = true;
                }
            }

            // Two input registers
            case OP_ADD,
            OP_MUL,
            OP_DIV,
            OP_SUB,
            OP_COMPARE,
            OP_ATAN2,
            OP_MOD: {
                if op[2] != 255 {
                    live[op[2]] = true;
                }
                if op[3] != 255 {
                    live[op[3]] = true;
                }
            }

            // Two input registers, and a choice
            case OP_MIN,
            OP_MAX,
            OP_AND,
            OP_OR: {
                switch stack_pop(stack) {
                    case CHOICE_LEFT: {
                        op[0] = OP_COPY; // argument is already in op[2]
                        if op[2] != 255 {
                            live[op[2]] = true;
                        }
                        if op[2] == op[1] { // skip reassignment
                            j += 2;
                            continue;
                        }
                    }
                    case CHOICE_RIGHT: {
                        op[0] = OP_COPY;
                        if op[3] != 255 {
                            live[op[3]] = true;
                        }
                        op[2] = op[3];
                        if op[2] == op[1] { // skip reassignment
                            j += 2;
                            continue;
                        }
                    }
                    default: { // should always be CHOICE_BOTH
                        if op[2] != 255 {
                            live[op[2]] = true;
                        }
                        if op[3] != 255 {
                            live[op[3]] = true;
                        }
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
    return atomicAdd(&tape_data.offset, chunk_size) + tape_data.base_len;
}

/// Undo an allocation
fn dealloc(chunk_size: u32) {
    atomicSub(&tape_data.offset, chunk_size);
}
