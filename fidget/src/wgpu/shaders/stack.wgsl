const STACK_SIZE: u32 = 8; // number of words in the stack

const CHOICE_LEFT: u32 = 1;
const CHOICE_RIGHT: u32 = 2;
const CHOICE_BOTH: u32 = 3;

// Lossy stack, which expects a set of pushes followed by a set of pops
// (mixing pushes and pops is not allowed)
struct Stack {
    cur_depth: u32,
    max_depth: u32,
    data: array<u32, STACK_SIZE>,
}

// Pushes a value to the stack, wrapping on overflow
fn stack_push(s: ptr<function, Stack>, v: u32) {
    let bit_pos = s.cur_depth * 2u;
    let word_offset = bit_pos % 32u;
    let word_index = (bit_pos / 32u) % STACK_SIZE;

    let mask = 3u << word_offset;
    s.data[word_index] &= ~mask;
    s.data[word_index] |= (v & 3u) << word_offset;

    s.cur_depth += 1u;
    s.max_depth = max(s.max_depth, s.cur_depth);
}

// Pops a value from the stack, returning 0xFFFFFFFF on underflow
fn stack_pop(s: ptr<function, Stack>) -> u32 {
    if s.cur_depth == 0u {
        return 0xFFFFFFFFu;
    }

    s.cur_depth -= 1u;
    if (s.cur_depth + STACK_SIZE * 16 == s.max_depth) {
        return 0xFFFFFFFFu;
    }

    let bit_pos = s.cur_depth * 2u;
    let word_offset = bit_pos % 32u;
    let word_index = (bit_pos / 32u) % STACK_SIZE;

    return (s.data[word_index] >> word_offset) & 3u;
}
