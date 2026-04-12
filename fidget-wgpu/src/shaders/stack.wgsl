const STACK_SIZE_WORDS: u32 = 32; // number of u32 words in the stack
const STACK_SIZE_BITS: u32 = STACK_SIZE_WORDS * 32u;
const STACK_SIZE_ITEMS: u32 = STACK_SIZE_WORDS * 16u;

const CHOICE_LEFT: u32 = 1;
const CHOICE_RIGHT: u32 = 2;
const CHOICE_BOTH: u32 = 3;

// Lossy stack, which expects a set of pushes followed by a set of pops
// (mixing pushes and pops is not allowed)
struct Stack {
    /// Current bitwise offset in the stack
    offset: u32,

    /// Number of items stored in the stack, (saturating at STACK_SIZE_ITEMS)
    valid_count: u32,

    /// Has a CHOICE_LEFT or CHOICE_RIGHT been pushed to the stack?
    has_choice: bool,

    /// Raw stack data
    data: array<u32, STACK_SIZE_WORDS>,
}

// Pushes a value to the stack, wrapping on overflow
fn stack_push(s: ptr<function, Stack>, v: u32) {
    let word_offset = s.offset % 32u;
    let word_index = s.offset / 32u;

    let mask = 3u << word_offset;
    s.data[word_index] &= ~mask;
    s.data[word_index] |= (v & 3u) << word_offset;

    s.has_choice |= (v == CHOICE_LEFT) || (v == CHOICE_RIGHT);

    s.offset = (s.offset + 2u) % STACK_SIZE_BITS;
    s.valid_count = min(s.valid_count + 1, STACK_SIZE_ITEMS);
}

// Pops a value from the stack, returning CHOICE_BOTH on underflow
fn stack_pop(s: ptr<function, Stack>) -> u32 {
    if s.valid_count == 0u {
        return CHOICE_BOTH;
    }
    s.valid_count -= 1;

    s.offset = (s.offset + STACK_SIZE_BITS - 2u) % STACK_SIZE_BITS;
    let word_offset = s.offset % 32u;
    let word_index = s.offset / 32u;

    return (s.data[word_index] >> word_offset) & 3u;
}
