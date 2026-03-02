// Column-per-workgroup shader (v21 — inline gradient normals)
//
// Single-dispatch architecture: depth finding + gradient normal computation.
// Each workgroup processes an entire XY sub-column (16x16 pixels) through all
// root Z layers, front-to-back. For each z16 slice:
//   Phase 1 (parallel): each thread evaluates its 4^3 interval + simplifies tape
//   Phase 2 (cooperative): all 64 threads work together on each ambiguous tile,
//     one voxel per thread (64 voxels in parallel), front-to-back by Z layer
// After all z16 slices, write_output computes gradient normals (forward-mode AD)
// for each surface pixel using tile16-simplified tapes.
//
// Must be combined with: opcode constants, REG_COUNT, TAPE_BUDGET,
// common.wgsl, stack.wgsl.

// == TapeData / TapeWord ==
// Non-atomic struct for read-only binding (WebGPU requires read_write for atomics)
struct TapeDataRO {
    offset: u32,
    base_len: u32,
    capacity: u32,
    data: array<TapeWord>,
}

struct TapeWord {
    op: u32,
    imm: u32,
}

const OP_JUMP: u32 = 0xFF;
const CHUNK_SIZE: u32 = 64u;

// Packed word encoding: upper 12 bits = status/z, lower 20 bits = tape offset
const STATUS_SHIFT: u32 = 20u;
const TAPE_OFFSET_MASK: u32 = 0xFFFFFu;

// tile4_tape encoding: bit 31 = tape is in workgroup_tapes (local), otherwise tape_data
const LOCAL_TAPE_BIT: u32 = 0x80000000u;

// Returned by simplify_tape_local when tape budget is exceeded
const SIMPLIFY_FAILED: u32 = 0xFFFFFFFFu;

// == Bindings ==
@group(1) @binding(0) var<storage, read> tape_data: TapeDataRO;
@group(1) @binding(1) var<storage, read> root_status: array<u32>;
@group(1) @binding(2) var<storage, read> tile64_zmin: array<u32>;
@group(1) @binding(3) var<storage, read> tile16_status: array<u32>;
@group(1) @binding(4) var<storage, read_write> workgroup_tapes: array<TapeWord>;
@group(1) @binding(5) var<storage, read_write> image_out: array<vec4f>;

// == Constants ==
const ROOT_EMPTY: u32 = 0u;
const ROOT_FILLED: u32 = 1u;
const ROOT_AMBIGUOUS: u32 = 2u;

const TILE_SIZE: u32 = 64u;

// == Shared memory ==
var<workgroup> tile4_zmin: array<atomic<u32>, 16>;
var<workgroup> voxel_zmin: array<atomic<u32>, 256>;
var<workgroup> tile4_tape: array<u32, 64>;  // Phase 1→2 communication
var<workgroup> wg_action: u32;               // uniform control flow decisions

// ============================================================================
// Interval Value type + operations (for 4^3 tile evaluation)
// ============================================================================

struct IntervalValue { v: vec2f, }

fn interval_is_nan(v: f32) -> bool {
    let u = bitcast<u32>(v);
    return ((u >> 23u) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u;
}
fn interval_has_nan(i: IntervalValue) -> bool {
    return interval_is_nan(i.v.x) || interval_is_nan(i.v.y);
}
fn interval_nan_i() -> IntervalValue { return IntervalValue(vec2f(nan_f32())); }
fn interval_contains(i: IntervalValue, val: f32) -> bool {
    return i.v[0] <= val && val <= i.v[1];
}
fn interval_build_imm(v: f32) -> IntervalValue { return IntervalValue(vec2f(v)); }

fn interval_op_neg(a: IntervalValue) -> IntervalValue { return IntervalValue(-a.v.yx); }
fn interval_op_abs(a: IntervalValue) -> IntervalValue {
    if a.v[0] < 0.0 {
        if a.v[1] > 0.0 { return IntervalValue(vec2f(0.0, max(a.v[1], -a.v[0]))); }
        else { return IntervalValue(vec2f(-a.v[1], -a.v[0])); }
    } else { return a; }
}
fn interval_op_recip(a: IntervalValue) -> IntervalValue {
    if a.v[0] > 0.0 || a.v[1] < 0.0 { return IntervalValue(vec2f(1.0 / a.v[1], 1.0 / a.v[0])); }
    else { return interval_nan_i(); }
}
fn interval_op_sqrt(a: IntervalValue) -> IntervalValue {
    if a.v[0] >= 0.0 { return IntervalValue(sqrt(a.v)); } else { return interval_nan_i(); }
}
fn interval_op_floor(a: IntervalValue) -> IntervalValue { return IntervalValue(floor(a.v)); }
fn interval_op_ceil(a: IntervalValue) -> IntervalValue { return IntervalValue(ceil(a.v)); }
fn interval_op_round(a: IntervalValue) -> IntervalValue { return IntervalValue(round(a.v)); }
fn interval_op_square(a: IntervalValue) -> IntervalValue {
    if a.v[1] < 0.0 { return IntervalValue(vec2f(a.v[1]*a.v[1], a.v[0]*a.v[0])); }
    else if a.v[0] > 0.0 { return IntervalValue(vec2f(a.v[0]*a.v[0], a.v[1]*a.v[1])); }
    else if interval_has_nan(a) { return interval_nan_i(); }
    else { let v = max(abs(a.v[0]), abs(a.v[1])); return IntervalValue(vec2f(0.0, v*v)); }
}
fn interval_op_sin(a: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) { return interval_nan_i(); } else { return IntervalValue(vec2f(-1.0, 1.0)); }
}
fn interval_op_cos(a: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) { return interval_nan_i(); } else { return IntervalValue(vec2f(-1.0, 1.0)); }
}
fn interval_op_tan(a: IntervalValue) -> IntervalValue {
    let size = a.v[1] - a.v[0];
    if size >= 3.14159265 { return interval_nan_i(); }
    let lo = tan(a.v[0]); let hi = tan(a.v[1]);
    if hi >= lo { return IntervalValue(vec2f(lo, hi)); } else { return interval_nan_i(); }
}
fn interval_op_asin(a: IntervalValue) -> IntervalValue {
    if a.v[0] < -1.0 || a.v[1] > 1.0 { return interval_nan_i(); } else { return IntervalValue(asin(a.v)); }
}
fn interval_op_acos(a: IntervalValue) -> IntervalValue {
    if a.v[0] < -1.0 || a.v[1] > 1.0 { return interval_nan_i(); } else { return IntervalValue(acos(a.v).yx); }
}
fn interval_op_atan(a: IntervalValue) -> IntervalValue { return IntervalValue(atan(a.v)); }
fn interval_op_exp(a: IntervalValue) -> IntervalValue { return IntervalValue(exp(a.v)); }
fn interval_op_log(a: IntervalValue) -> IntervalValue {
    if a.v[0] < 0.0 { return interval_nan_i(); } else { return IntervalValue(log(a.v)); }
}
fn interval_op_not(a: IntervalValue) -> IntervalValue {
    if !interval_contains(a, 0.0) && !interval_has_nan(a) { return IntervalValue(vec2f(0.0)); }
    else if a.v[0] == 0.0 && a.v[1] == 0.0 { return IntervalValue(vec2f(1.0)); }
    else { return IntervalValue(vec2f(0.0, 1.0)); }
}

fn interval_op_add(a: IntervalValue, b: IntervalValue) -> IntervalValue { return IntervalValue(a.v + b.v); }
fn interval_op_sub(a: IntervalValue, b: IntervalValue) -> IntervalValue { return IntervalValue(a.v - b.v.yx); }
fn interval_op_mul(a: IntervalValue, b: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { return interval_nan_i(); }
    let ab = a.v * b.v; let cd = a.v.yx * b.v;
    return IntervalValue(vec2f(min(min(ab[0],ab[1]),min(cd[0],cd[1])), max(max(ab[0],ab[1]),max(cd[0],cd[1]))));
}
fn interval_op_div(a: IntervalValue, b: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) || interval_contains(b, 0.0) { return interval_nan_i(); }
    let ab = a.v / b.v; let cd = a.v.yx / b.v;
    return IntervalValue(vec2f(min(min(ab[0],ab[1]),min(cd[0],cd[1])), max(max(ab[0],ab[1]),max(cd[0],cd[1]))));
}
fn interval_op_compare(a: IntervalValue, b: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { return interval_nan_i(); }
    else if a.v[1] < b.v[0] { return IntervalValue(vec2f(-1.0)); }
    else if a.v[0] > b.v[1] { return IntervalValue(vec2f(1.0)); }
    else if a.v[0] == a.v[1] && b.v[0] == b.v[1] && a.v[0] == b.v[0] { return IntervalValue(vec2f(0.0)); }
    else { return IntervalValue(vec2f(-1.0, 1.0)); }
}
fn interval_op_atan2(a: IntervalValue, b: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { return interval_nan_i(); }
    return IntervalValue(vec2f(-3.141592654, 3.141592654));
}
fn interval_rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs; if r < 0.0 { return r + abs(rhs); } else { return r; }
}
fn interval_op_mod(a: IntervalValue, b: IntervalValue) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { return interval_nan_i(); }
    else if b.v[0] == b.v[1] && b.v[0] > 0.0 {
        let lo = a.v[0] / b.v[0]; let hi = a.v[1] / b.v[0];
        if lo != floor(lo) && floor(lo) == floor(hi) {
            return IntervalValue(vec2f(interval_rem_euclid(a.v[0], b.v[0]), interval_rem_euclid(a.v[1], b.v[0])));
        } else { return IntervalValue(vec2f(0.0, abs(b.v[0]))); }
    } else { return IntervalValue(vec2f(0.0, max(abs(b.v[0]), abs(b.v[1])))); }
}
fn interval_op_min(a: IntervalValue, b: IntervalValue, stack: ptr<function, Stack>) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { stack_push(stack, CHOICE_BOTH); return interval_nan_i(); }
    else if a.v[1] < b.v[0] { stack_push(stack, CHOICE_LEFT); return a; }
    else if b.v[1] < a.v[0] { stack_push(stack, CHOICE_RIGHT); return b; }
    else { stack_push(stack, CHOICE_BOTH); return IntervalValue(min(a.v, b.v)); }
}
fn interval_op_max(a: IntervalValue, b: IntervalValue, stack: ptr<function, Stack>) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { stack_push(stack, CHOICE_BOTH); return interval_nan_i(); }
    else if a.v[0] > b.v[1] { stack_push(stack, CHOICE_LEFT); return a; }
    else if b.v[0] > a.v[1] { stack_push(stack, CHOICE_RIGHT); return b; }
    else { stack_push(stack, CHOICE_BOTH); return IntervalValue(max(a.v, b.v)); }
}
fn interval_op_and(a: IntervalValue, b: IntervalValue, stack: ptr<function, Stack>) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { stack_push(stack, CHOICE_BOTH); return interval_nan_i(); }
    else if a.v[0] == 0.0 && a.v[1] == 0.0 { stack_push(stack, CHOICE_LEFT); return IntervalValue(vec2f(0.0)); }
    else if !interval_contains(a, 0.0) { stack_push(stack, CHOICE_RIGHT); return b; }
    else { stack_push(stack, CHOICE_BOTH); return IntervalValue(vec2f(min(b.v[0], 0.0), max(b.v[1], 0.0))); }
}
fn interval_op_or(a: IntervalValue, b: IntervalValue, stack: ptr<function, Stack>) -> IntervalValue {
    if interval_has_nan(a) || interval_has_nan(b) { stack_push(stack, CHOICE_BOTH); return interval_nan_i(); }
    else if !interval_contains(a, 0.0) { stack_push(stack, CHOICE_LEFT); return a; }
    else if a.v[0] == 0.0 && a.v[1] == 0.0 { stack_push(stack, CHOICE_RIGHT); return b; }
    else { stack_push(stack, CHOICE_BOTH); return IntervalValue(vec2f(min(a.v[0], b.v[0]), max(a.v[1], b.v[1]))); }
}

fn interval_inputs(tile_corner: vec3u, ts: u32) -> array<IntervalValue, 3> {
    let cp = tile_corner * ts;
    let ix = IntervalValue(vec2f(f32(cp.x), f32(cp.x + ts)));
    let iy = IntervalValue(vec2f(f32(cp.y), f32(cp.y + ts)));
    let iz = IntervalValue(vec2f(f32(cp.z), f32(cp.z + ts)));
    var t = array(IntervalValue(), IntervalValue(), IntervalValue(), IntervalValue());
    for (var i = 0; i < 4; i++) {
        t[i] = interval_op_add(
            interval_op_add(interval_op_mul(interval_build_imm(config.mat[0][i]), ix),
                            interval_op_mul(interval_build_imm(config.mat[1][i]), iy)),
            interval_op_add(interval_op_mul(interval_build_imm(config.mat[2][i]), iz),
                            interval_build_imm(config.mat[3][i])));
    }
    var m = array(IntervalValue(), IntervalValue(), IntervalValue());
    if config.axes.x < 3 { m[config.axes.x] = interval_op_div(t[0], t[3]); }
    if config.axes.y < 3 { m[config.axes.y] = interval_op_div(t[1], t[3]); }
    if config.axes.z < 3 { m[config.axes.z] = interval_op_div(t[2], t[3]); }
    return m;
}

// Interval interpreter — reads from tape_data (root tapes)
struct IntervalTapeResult { value: IntervalValue, pos: u32, count: u32, }

fn run_tape_interval(start: u32, inputs: array<IntervalValue, 3>, stack: ptr<function, Stack>) -> IntervalTapeResult {
    var i = start; var count = 0u;
    var reg: array<IntervalValue, REG_COUNT>;
    var lhs = IntervalValue(); var rhs = IntervalValue();
    var out = IntervalTapeResult(interval_build_imm(nan_f32()), 0, 0);
    while true {
        count += 1;
        let word = tape_data.data[i];
        let op = unpack4xU8(word.op);
        let imm_u = word.imm;
        let imm_v = interval_build_imm(bitcast<f32>(imm_u));
        if op[2] == 255 { lhs = imm_v; } else { lhs = reg[op[2]]; }
        if op[3] == 255 { rhs = imm_v; } else { rhs = reg[op[3]]; }
        var tmp = interval_build_imm(0.0);
        i += 1;
        switch op[0] {
            case OP_OUTPUT: { out.value = reg[op[1]]; continue; }
            case OP_INPUT: { tmp = inputs[imm_u]; }
            case OP_COPY: { tmp = lhs; }
            case OP_NEG: { tmp = interval_op_neg(lhs); } case OP_ABS: { tmp = interval_op_abs(lhs); }
            case OP_RECIP: { tmp = interval_op_recip(lhs); } case OP_SQRT: { tmp = interval_op_sqrt(lhs); }
            case OP_SQUARE: { tmp = interval_op_square(lhs); } case OP_FLOOR: { tmp = interval_op_floor(lhs); }
            case OP_CEIL: { tmp = interval_op_ceil(lhs); } case OP_ROUND: { tmp = interval_op_round(lhs); }
            case OP_SIN: { tmp = interval_op_sin(lhs); } case OP_COS: { tmp = interval_op_cos(lhs); }
            case OP_TAN: { tmp = interval_op_tan(lhs); } case OP_ASIN: { tmp = interval_op_asin(lhs); }
            case OP_ACOS: { tmp = interval_op_acos(lhs); } case OP_ATAN: { tmp = interval_op_atan(lhs); }
            case OP_EXP: { tmp = interval_op_exp(lhs); } case OP_LN: { tmp = interval_op_log(lhs); }
            case OP_NOT: { tmp = interval_op_not(lhs); }
            case OP_ADD: { tmp = interval_op_add(lhs, rhs); } case OP_MUL: { tmp = interval_op_mul(lhs, rhs); }
            case OP_DIV: { tmp = interval_op_div(lhs, rhs); } case OP_SUB: { tmp = interval_op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = interval_op_compare(lhs, rhs); }
            case OP_ATAN2: { tmp = interval_op_atan2(lhs, rhs); } case OP_MOD: { tmp = interval_op_mod(lhs, rhs); }
            case OP_MIN: { tmp = interval_op_min(lhs, rhs, stack); } case OP_MAX: { tmp = interval_op_max(lhs, rhs, stack); }
            case OP_AND: { tmp = interval_op_and(lhs, rhs, stack); } case OP_OR: { tmp = interval_op_or(lhs, rhs, stack); }
            case OP_LOAD, OP_STORE: { return out; }
            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu { out.pos = i; out.count = count; return out; }
                else if imm_u == 0u { continue; } else { i = imm_u; continue; }
            }
            default: { return out; }
        }
        reg[op[1]] = tmp;
    }
    return out;
}

// ============================================================================
// Float Value type + operations (for voxel hit testing)
// ============================================================================

struct FloatValue { v: f32, }
fn float_build_imm(v: f32) -> FloatValue { return FloatValue(v); }
fn float_op_neg(a: FloatValue) -> FloatValue { return FloatValue(-a.v); }
fn float_op_abs(a: FloatValue) -> FloatValue { return FloatValue(abs(a.v)); }
fn float_op_recip(a: FloatValue) -> FloatValue { return FloatValue(1.0/a.v); }
fn float_op_sqrt(a: FloatValue) -> FloatValue { return FloatValue(sqrt(a.v)); }
fn float_op_square(a: FloatValue) -> FloatValue { return FloatValue(a.v*a.v); }
fn float_op_floor(a: FloatValue) -> FloatValue { return FloatValue(floor(a.v)); }
fn float_op_ceil(a: FloatValue) -> FloatValue { return FloatValue(ceil(a.v)); }
fn float_op_round(a: FloatValue) -> FloatValue { return FloatValue(round(a.v)); }
fn float_op_sin(a: FloatValue) -> FloatValue { return FloatValue(sin(a.v)); }
fn float_op_cos(a: FloatValue) -> FloatValue { return FloatValue(cos(a.v)); }
fn float_op_tan(a: FloatValue) -> FloatValue { return FloatValue(tan(a.v)); }
fn float_op_asin(a: FloatValue) -> FloatValue { return FloatValue(asin(a.v)); }
fn float_op_acos(a: FloatValue) -> FloatValue { return FloatValue(acos(a.v)); }
fn float_op_atan(a: FloatValue) -> FloatValue { return FloatValue(atan(a.v)); }
fn float_op_exp(a: FloatValue) -> FloatValue { return FloatValue(exp(a.v)); }
fn float_op_log(a: FloatValue) -> FloatValue { return FloatValue(log(a.v)); }
fn float_op_not(a: FloatValue) -> FloatValue { return FloatValue(f32(a.v != 0.0)); }
fn float_op_add(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(a.v+b.v); }
fn float_op_sub(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(a.v-b.v); }
fn float_op_mul(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(a.v*b.v); }
fn float_op_div(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(a.v/b.v); }
fn float_op_min(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(min(a.v,b.v)); }
fn float_op_max(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(max(a.v,b.v)); }
fn float_op_atan2(a: FloatValue, b: FloatValue) -> FloatValue { return FloatValue(atan2(a.v,b.v)); }
fn float_op_and(a: FloatValue, b: FloatValue) -> FloatValue {
    if a.v == 0.0 { return a; } else { return b; }
}
fn float_op_or(a: FloatValue, b: FloatValue) -> FloatValue {
    if a.v != 0.0 { return a; } else { return b; }
}
fn float_op_compare(a: FloatValue, b: FloatValue) -> FloatValue {
    if a.v < b.v { return FloatValue(-1.0); }
    else if a.v > b.v { return FloatValue(1.0); }
    else if a.v == b.v { return FloatValue(0.0); }
    else { return FloatValue(nan_f32()); }
}
fn float_op_mod(a: FloatValue, b: FloatValue) -> FloatValue {
    var o = a.v % b.v; o -= b.v * min(0.0, floor(o / b.v)); return FloatValue(o);
}

fn float_transformed_inputs(fx: f32, fy: f32, fz: f32) -> array<FloatValue, 3> {
    let ix = FloatValue(fx); let iy = FloatValue(fy); let iz = FloatValue(fz);
    var t = array(FloatValue(), FloatValue(), FloatValue(), FloatValue());
    for (var i = 0; i < 4; i++) {
        t[i] = float_op_add(
            float_op_add(float_op_mul(float_build_imm(config.mat[0][i]), ix),
                         float_op_mul(float_build_imm(config.mat[1][i]), iy)),
            float_op_add(float_op_mul(float_build_imm(config.mat[2][i]), iz),
                         float_build_imm(config.mat[3][i])));
    }
    var m = array(FloatValue(), FloatValue(), FloatValue());
    if config.axes.x < 3 { m[config.axes.x] = float_op_div(t[0], t[3]); }
    if config.axes.y < 3 { m[config.axes.y] = float_op_div(t[1], t[3]); }
    if config.axes.z < 3 { m[config.axes.z] = float_op_div(t[2], t[3]); }
    return m;
}

// ============================================================================
// Local tape simplification (writes to workgroup_tapes)
// ============================================================================

// Simplify a tape from tape_data into a per-thread region of workgroup_tapes.
// Returns the start index in workgroup_tapes, or SIMPLIFY_FAILED if budget exceeded.
fn simplify_tape_local(
    end: u32,
    count: u32,
    stack: ptr<function, Stack>,
    tape_base: u32,
    tape_budget: u32,
) -> u32 {
    let chunk_size = CHUNK_SIZE;
    var local_offset = chunk_size;  // first chunk at offset 0
    var chunk_start = tape_base;
    var j = tape_base + chunk_size - 1;

    var live: array<bool, REG_COUNT>;
    var i = end;
    while true {
        i -= 1; j -= 1;
        let word = tape_data.data[i];
        var op = unpack4xU8(word.op);
        let imm_u = word.imm;

        if op[0] == OP_JUMP {
            if imm_u == 0xFFFFFFFFu {
                workgroup_tapes[j] = TapeWord(OP_JUMP, 0xFFFFFFFFu);
                continue;
            } else if imm_u == 0u {
                workgroup_tapes[j] = TapeWord(OP_JUMP, 0);
                return j;
            } else {
                i = imm_u + 1; j += 1; continue;
            }
        }

        if j == chunk_start {
            chunk_start = tape_base + local_offset;
            local_offset += chunk_size;
            if local_offset > tape_budget { return SIMPLIFY_FAILED; }
            let nj = chunk_start + chunk_size - 1;
            workgroup_tapes[j] = TapeWord(OP_JUMP, nj - 1);
            workgroup_tapes[nj] = TapeWord(OP_JUMP, j + 1);
            j = nj - 1;
        }

        if op[0] == OP_OUTPUT {
            live[op[1]] = true;
        } else if !live[op[1]] {
            switch op[0] {
                case OP_MIN, OP_MAX, OP_AND, OP_OR: { stack_pop(stack); }
                default: {}
            }
            j += 1; continue;
        } else {
            live[op[1]] = false;
        }

        switch op[0] {
            case OP_OUTPUT: {}
            case OP_INPUT: {}
            case OP_COPY, OP_NEG, OP_ABS, OP_RECIP, OP_SQRT, OP_SQUARE,
                 OP_FLOOR, OP_CEIL, OP_ROUND, OP_SIN, OP_COS, OP_TAN,
                 OP_ASIN, OP_ACOS, OP_ATAN, OP_EXP, OP_LN, OP_NOT: {
                if op[2] != 255 { live[op[2]] = true; }
            }
            case OP_ADD, OP_MUL, OP_DIV, OP_SUB, OP_COMPARE, OP_ATAN2, OP_MOD: {
                if op[2] != 255 { live[op[2]] = true; }
                if op[3] != 255 { live[op[3]] = true; }
            }
            case OP_MIN, OP_MAX, OP_AND, OP_OR: {
                switch stack_pop(stack) {
                    case CHOICE_LEFT: {
                        op[0] = OP_COPY;
                        if op[2] != 255 { live[op[2]] = true; }
                        if op[2] == op[1] { j += 1; continue; }
                    }
                    case CHOICE_RIGHT: {
                        op[0] = OP_COPY;
                        if op[3] != 255 { live[op[3]] = true; }
                        op[2] = op[3];
                        if op[2] == op[1] { j += 1; continue; }
                    }
                    default: {
                        if op[2] != 255 { live[op[2]] = true; }
                        if op[3] != 255 { live[op[3]] = true; }
                    }
                }
            }
            case OP_JUMP: {}
            case OP_LOAD, OP_STORE: { return SIMPLIFY_FAILED; }
            default: { return SIMPLIFY_FAILED; }
        }
        workgroup_tapes[j] = TapeWord(pack4xU8(op), imm_u);
    }
    return SIMPLIFY_FAILED;
}

// Float interpreter — reads from tape_data (for z16 tape fallback)
fn run_tape_float_root(start: u32, inputs: array<FloatValue, 3>) -> FloatValue {
    var i = start; var reg: array<FloatValue, REG_COUNT>;
    var lhs = FloatValue(); var rhs = FloatValue();
    var out = float_build_imm(nan_f32());
    while true {
        let word = tape_data.data[i]; let op = unpack4xU8(word.op);
        let imm_u = word.imm; let imm_v = float_build_imm(bitcast<f32>(imm_u));
        if op[2] == 255 { lhs = imm_v; } else { lhs = reg[op[2]]; }
        if op[3] == 255 { rhs = imm_v; } else { rhs = reg[op[3]]; }
        var tmp = float_build_imm(0.0); i += 1;
        switch op[0] {
            case OP_OUTPUT: { out = reg[op[1]]; continue; }
            case OP_INPUT: { tmp = inputs[imm_u]; } case OP_COPY: { tmp = lhs; }
            case OP_NEG: { tmp = float_op_neg(lhs); } case OP_ABS: { tmp = float_op_abs(lhs); }
            case OP_RECIP: { tmp = float_op_recip(lhs); } case OP_SQRT: { tmp = float_op_sqrt(lhs); }
            case OP_SQUARE: { tmp = float_op_square(lhs); } case OP_FLOOR: { tmp = float_op_floor(lhs); }
            case OP_CEIL: { tmp = float_op_ceil(lhs); } case OP_ROUND: { tmp = float_op_round(lhs); }
            case OP_SIN: { tmp = float_op_sin(lhs); } case OP_COS: { tmp = float_op_cos(lhs); }
            case OP_TAN: { tmp = float_op_tan(lhs); } case OP_ASIN: { tmp = float_op_asin(lhs); }
            case OP_ACOS: { tmp = float_op_acos(lhs); } case OP_ATAN: { tmp = float_op_atan(lhs); }
            case OP_EXP: { tmp = float_op_exp(lhs); } case OP_LN: { tmp = float_op_log(lhs); }
            case OP_NOT: { tmp = float_op_not(lhs); }
            case OP_ADD: { tmp = float_op_add(lhs, rhs); } case OP_MUL: { tmp = float_op_mul(lhs, rhs); }
            case OP_DIV: { tmp = float_op_div(lhs, rhs); } case OP_SUB: { tmp = float_op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = float_op_compare(lhs, rhs); }
            case OP_ATAN2: { tmp = float_op_atan2(lhs, rhs); } case OP_MOD: { tmp = float_op_mod(lhs, rhs); }
            case OP_MIN: { tmp = float_op_min(lhs, rhs); } case OP_MAX: { tmp = float_op_max(lhs, rhs); }
            case OP_AND: { tmp = float_op_and(lhs, rhs); } case OP_OR: { tmp = float_op_or(lhs, rhs); }
            case OP_LOAD, OP_STORE: { return out; }
            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu { return out; }
                else if imm_u == 0u { continue; } else { i = imm_u; continue; }
            }
            default: { return out; }
        }
        reg[op[1]] = tmp;
    }
    return out;
}

// Float interpreter — reads from workgroup_tapes (for locally simplified tapes)
fn run_tape_float(start: u32, inputs: array<FloatValue, 3>) -> FloatValue {
    var i = start; var reg: array<FloatValue, REG_COUNT>;
    var lhs = FloatValue(); var rhs = FloatValue();
    var out = float_build_imm(nan_f32());
    while true {
        let word = workgroup_tapes[i]; let op = unpack4xU8(word.op);
        let imm_u = word.imm; let imm_v = float_build_imm(bitcast<f32>(imm_u));
        if op[2] == 255 { lhs = imm_v; } else { lhs = reg[op[2]]; }
        if op[3] == 255 { rhs = imm_v; } else { rhs = reg[op[3]]; }
        var tmp = float_build_imm(0.0); i += 1;
        switch op[0] {
            case OP_OUTPUT: { out = reg[op[1]]; continue; }
            case OP_INPUT: { tmp = inputs[imm_u]; } case OP_COPY: { tmp = lhs; }
            case OP_NEG: { tmp = float_op_neg(lhs); } case OP_ABS: { tmp = float_op_abs(lhs); }
            case OP_RECIP: { tmp = float_op_recip(lhs); } case OP_SQRT: { tmp = float_op_sqrt(lhs); }
            case OP_SQUARE: { tmp = float_op_square(lhs); } case OP_FLOOR: { tmp = float_op_floor(lhs); }
            case OP_CEIL: { tmp = float_op_ceil(lhs); } case OP_ROUND: { tmp = float_op_round(lhs); }
            case OP_SIN: { tmp = float_op_sin(lhs); } case OP_COS: { tmp = float_op_cos(lhs); }
            case OP_TAN: { tmp = float_op_tan(lhs); } case OP_ASIN: { tmp = float_op_asin(lhs); }
            case OP_ACOS: { tmp = float_op_acos(lhs); } case OP_ATAN: { tmp = float_op_atan(lhs); }
            case OP_EXP: { tmp = float_op_exp(lhs); } case OP_LN: { tmp = float_op_log(lhs); }
            case OP_NOT: { tmp = float_op_not(lhs); }
            case OP_ADD: { tmp = float_op_add(lhs, rhs); } case OP_MUL: { tmp = float_op_mul(lhs, rhs); }
            case OP_DIV: { tmp = float_op_div(lhs, rhs); } case OP_SUB: { tmp = float_op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = float_op_compare(lhs, rhs); }
            case OP_ATAN2: { tmp = float_op_atan2(lhs, rhs); } case OP_MOD: { tmp = float_op_mod(lhs, rhs); }
            case OP_MIN: { tmp = float_op_min(lhs, rhs); } case OP_MAX: { tmp = float_op_max(lhs, rhs); }
            case OP_AND: { tmp = float_op_and(lhs, rhs); } case OP_OR: { tmp = float_op_or(lhs, rhs); }
            case OP_LOAD, OP_STORE: { return out; }
            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu { return out; }
                else if imm_u == 0u { continue; } else { i = imm_u; continue; }
            }
            default: { return out; }
        }
        reg[op[1]] = tmp;
    }
    return out;
}

// ============================================================================
// Gradient Value type + operations (for normal computation at surface hits)
// Forward-mode AD: vec4f.xyz = gradient, vec4f.w = value
// ============================================================================

struct GradValue { v: vec4f, }
fn grad_build_imm(v: f32) -> GradValue { return GradValue(vec4f(0.0, 0.0, 0.0, v)); }
fn grad_op_neg(a: GradValue) -> GradValue { return GradValue(-a.v); }
fn grad_op_abs(a: GradValue) -> GradValue {
    if a.v.w < 0.0 { return GradValue(-a.v); } else { return a; }
}
fn grad_op_recip(a: GradValue) -> GradValue {
    let d = -a.v.w * a.v.w;
    return GradValue(vec4f(a.v.xyz / d, 1.0 / a.v.w));
}
fn grad_op_sqrt(a: GradValue) -> GradValue {
    let v = sqrt(a.v.w);
    return GradValue(vec4f(a.v.xyz / (2.0 * v), v));
}
fn grad_op_floor(a: GradValue) -> GradValue { return grad_build_imm(floor(a.v.w)); }
fn grad_op_ceil(a: GradValue) -> GradValue { return grad_build_imm(ceil(a.v.w)); }
fn grad_op_round(a: GradValue) -> GradValue { return grad_build_imm(round(a.v.w)); }
fn grad_op_square(a: GradValue) -> GradValue { return grad_op_mul(a, a); }
fn grad_op_sin(a: GradValue) -> GradValue {
    return GradValue(vec4f(a.v.xyz * cos(a.v.w), sin(a.v.w)));
}
fn grad_op_cos(a: GradValue) -> GradValue {
    return GradValue(vec4f(a.v.xyz * -sin(a.v.w), cos(a.v.w)));
}
fn grad_op_tan(a: GradValue) -> GradValue {
    let c = cos(a.v.w);
    return GradValue(vec4f(a.v.xyz / (c * c), tan(a.v.w)));
}
fn grad_op_asin(a: GradValue) -> GradValue {
    let r = sqrt(1.0 - a.v.w * a.v.w);
    return GradValue(vec4f(a.v.xyz / r, asin(a.v.w)));
}
fn grad_op_acos(a: GradValue) -> GradValue {
    let r = sqrt(1.0 - a.v.w * a.v.w);
    return GradValue(vec4f(-a.v.xyz / r, acos(a.v.w)));
}
fn grad_op_atan(a: GradValue) -> GradValue {
    let r = a.v.w * a.v.w + 1.0;
    return GradValue(vec4f(a.v.xyz / r, atan(a.v.w)));
}
fn grad_op_exp(a: GradValue) -> GradValue {
    let v = exp(a.v.w);
    return GradValue(vec4f(a.v.xyz * v, v));
}
fn grad_op_log(a: GradValue) -> GradValue {
    return GradValue(vec4f(a.v.xyz / a.v.w, log(a.v.w)));
}
fn grad_op_not(a: GradValue) -> GradValue { return grad_build_imm(f32(a.v.w == 0.0)); }
fn grad_op_add(a: GradValue, b: GradValue) -> GradValue { return GradValue(a.v + b.v); }
fn grad_op_sub(a: GradValue, b: GradValue) -> GradValue { return GradValue(a.v - b.v); }
fn grad_op_mul(a: GradValue, b: GradValue) -> GradValue {
    return GradValue(vec4f(a.v.xyz * b.v.w + b.v.xyz * a.v.w, a.v.w * b.v.w));
}
fn grad_op_div(a: GradValue, b: GradValue) -> GradValue {
    let d = b.v.w * b.v.w;
    return GradValue(vec4f((b.v.w * a.v.xyz - a.v.w * b.v.xyz) / d, a.v.w / b.v.w));
}
fn grad_op_compare(a: GradValue, b: GradValue) -> GradValue {
    if a.v.w < b.v.w { return grad_build_imm(-1.0); }
    else if a.v.w > b.v.w { return grad_build_imm(1.0); }
    else if a.v.w == b.v.w { return grad_build_imm(0.0); }
    else { return grad_build_imm(nan_f32()); }
}
fn grad_op_atan2(a: GradValue, b: GradValue) -> GradValue {
    let d = b.v.w * b.v.w + a.v.w * a.v.w;
    return GradValue(vec4f((b.v.w * a.v.xyz - a.v.w * b.v.xyz) / d, atan2(a.v.w, b.v.w)));
}
fn grad_rem_euclid(a: f32, b: f32) -> f32 {
    let r = a % b; if r < 0.0 { return r + abs(b); } else { return r; }
}
fn grad_div_euclid(a: f32, b: f32) -> f32 {
    let q = trunc(a / b);
    if a % b < 0.0 { if b > 0.0 { return q - 1.0; } else { return q + 1.0; } }
    else { return q; }
}
fn grad_op_mod(a: GradValue, b: GradValue) -> GradValue {
    let e = grad_div_euclid(a.v.w, b.v.w);
    return GradValue(vec4f(a.v.xyz - b.v.xyz * e, grad_rem_euclid(a.v.w, b.v.w)));
}
fn grad_op_min(a: GradValue, b: GradValue) -> GradValue {
    if a.v.w < b.v.w { return a; } else { return b; }
}
fn grad_op_max(a: GradValue, b: GradValue) -> GradValue {
    if a.v.w > b.v.w { return a; } else { return b; }
}
fn grad_op_and(a: GradValue, b: GradValue) -> GradValue {
    if a.v.w == 0.0 { return a; } else { return b; }
}
fn grad_op_or(a: GradValue, b: GradValue) -> GradValue {
    if a.v.w != 0.0 { return a; } else { return b; }
}

fn grad_transformed_inputs(fx: f32, fy: f32, fz: f32) -> array<GradValue, 3> {
    let ix = GradValue(vec4f(1.0, 0.0, 0.0, fx));
    let iy = GradValue(vec4f(0.0, 1.0, 0.0, fy));
    let iz = GradValue(vec4f(0.0, 0.0, 1.0, fz));
    var t = array(GradValue(), GradValue(), GradValue(), GradValue());
    for (var i = 0; i < 4; i++) {
        t[i] = grad_op_add(
            grad_op_add(grad_op_mul(grad_build_imm(config.mat[0][i]), ix),
                        grad_op_mul(grad_build_imm(config.mat[1][i]), iy)),
            grad_op_add(grad_op_mul(grad_build_imm(config.mat[2][i]), iz),
                        grad_build_imm(config.mat[3][i])));
    }
    var m = array(GradValue(), GradValue(), GradValue());
    if config.axes.x < 3 { m[config.axes.x] = grad_op_div(t[0], t[3]); }
    if config.axes.y < 3 { m[config.axes.y] = grad_op_div(t[1], t[3]); }
    if config.axes.z < 3 { m[config.axes.z] = grad_op_div(t[2], t[3]); }
    return m;
}

// Gradient interpreter — reads from tape_data (for z16 tape fallback)
fn run_tape_grad_root(start: u32, inputs: array<GradValue, 3>) -> GradValue {
    var i = start; var reg: array<GradValue, REG_COUNT>;
    var lhs = GradValue(); var rhs = GradValue();
    var out = grad_build_imm(nan_f32());
    while true {
        let word = tape_data.data[i]; let op = unpack4xU8(word.op);
        let imm_u = word.imm; let imm_v = grad_build_imm(bitcast<f32>(imm_u));
        if op[2] == 255 { lhs = imm_v; } else { lhs = reg[op[2]]; }
        if op[3] == 255 { rhs = imm_v; } else { rhs = reg[op[3]]; }
        var tmp = grad_build_imm(0.0); i += 1;
        switch op[0] {
            case OP_OUTPUT: { out = reg[op[1]]; continue; }
            case OP_INPUT: { tmp = inputs[imm_u]; } case OP_COPY: { tmp = lhs; }
            case OP_NEG: { tmp = grad_op_neg(lhs); } case OP_ABS: { tmp = grad_op_abs(lhs); }
            case OP_RECIP: { tmp = grad_op_recip(lhs); } case OP_SQRT: { tmp = grad_op_sqrt(lhs); }
            case OP_SQUARE: { tmp = grad_op_square(lhs); } case OP_FLOOR: { tmp = grad_op_floor(lhs); }
            case OP_CEIL: { tmp = grad_op_ceil(lhs); } case OP_ROUND: { tmp = grad_op_round(lhs); }
            case OP_SIN: { tmp = grad_op_sin(lhs); } case OP_COS: { tmp = grad_op_cos(lhs); }
            case OP_TAN: { tmp = grad_op_tan(lhs); } case OP_ASIN: { tmp = grad_op_asin(lhs); }
            case OP_ACOS: { tmp = grad_op_acos(lhs); } case OP_ATAN: { tmp = grad_op_atan(lhs); }
            case OP_EXP: { tmp = grad_op_exp(lhs); } case OP_LN: { tmp = grad_op_log(lhs); }
            case OP_NOT: { tmp = grad_op_not(lhs); }
            case OP_ADD: { tmp = grad_op_add(lhs, rhs); } case OP_MUL: { tmp = grad_op_mul(lhs, rhs); }
            case OP_DIV: { tmp = grad_op_div(lhs, rhs); } case OP_SUB: { tmp = grad_op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = grad_op_compare(lhs, rhs); }
            case OP_ATAN2: { tmp = grad_op_atan2(lhs, rhs); } case OP_MOD: { tmp = grad_op_mod(lhs, rhs); }
            case OP_MIN: { tmp = grad_op_min(lhs, rhs); } case OP_MAX: { tmp = grad_op_max(lhs, rhs); }
            case OP_AND: { tmp = grad_op_and(lhs, rhs); } case OP_OR: { tmp = grad_op_or(lhs, rhs); }
            case OP_LOAD, OP_STORE: { return out; }
            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu { return out; }
                else if imm_u == 0u { continue; } else { i = imm_u; continue; }
            }
            default: { return out; }
        }
        reg[op[1]] = tmp;
    }
    return out;
}

// ============================================================================
// Main entry point
// ============================================================================

@compute @workgroup_size(4, 4, 4)
fn column_main(
    @builtin(workgroup_id) _workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    // _workgroup_id.x is uniform per WGSL spec (same for all invocations in workgroup)
    let size_tiles = config.render_size / TILE_SIZE;
    let total_work_items = size_tiles.x * size_tiles.y * 16u;
    let stride = num_workgroups.x;

    var work_item = _workgroup_id.x;  // uniform: derived from uniform builtin
    while work_item < total_work_items {
        let root_xy_index = work_item / 16u;
        let sub_col = work_item % 16u;
        let root_x = root_xy_index % size_tiles.x;
        let root_y = root_xy_index / size_tiles.x;
        let sub_col_x = sub_col % 4u;
        let sub_col_y = sub_col / 4u;

        // Read initial occlusion from filled root tiles (computed by root pass)
        let initial_packed = tile64_zmin[root_x + root_y * size_tiles.x];
        let initial_z = initial_packed >> 20u;

        // Initialize shared memory cooperatively, seeded with initial occlusion
        for (var k = local_index; k < 16u; k += 64u) {
            atomicStore(&tile4_zmin[k], initial_z);
        }
        for (var k = local_index; k < 256u; k += 64u) {
            atomicStore(&voxel_zmin[k], initial_z);
        }
        workgroupBarrier();

        // Loop over root Z layers, front-to-back (highest Z first)
        for (var root_z_rev = 0u; root_z_rev < size_tiles.z; root_z_rev++) {
            let root_z = size_tiles.z - 1u - root_z_rev;
            let root_z_max = root_z * TILE_SIZE + TILE_SIZE - 1u;

            // Compute uniform action on thread 0:
            //   0 = skip (all occluded or empty)
            //   1 = filled
            //   2 = ambiguous
            // atomicLoad from workgroup vars is always non-uniform per WGSL spec,
            // so we use workgroupUniformLoad to make the decision uniform.
            if local_index == 0u {
                var action = 2u;
                // Occlusion check
                var all_occ = true;
                for (var k = 0u; k < 16u; k++) {
                    if atomicLoad(&tile4_zmin[k]) < root_z_max {
                        all_occ = false;
                        break;
                    }
                }
                if all_occ {
                    action = 0u;
                } else {
                    let root_idx = root_x + root_y * size_tiles.x
                        + root_z * size_tiles.x * size_tiles.y;
                    let status = root_status[root_idx] >> STATUS_SHIFT;
                    if status == ROOT_EMPTY {
                        action = 0u;
                    } else if status == ROOT_FILLED {
                        action = 1u;
                    }
                }
                wg_action = action;
            }
            let action = workgroupUniformLoad(&wg_action);

            if action == 0u { continue; }

            if action == 1u {
                process_filled_root(root_z, local_index);
                workgroupBarrier();
                continue;
            }

            // AMBIGUOUS
            process_ambiguous_root(root_x, root_y, root_z, sub_col_x, sub_col_y,
                                   local_id, local_index, _workgroup_id.x);
        }

        // Write results for remaining pixels (filled tiles) with gradient normals
        workgroupBarrier(); // ensure all Phase 2 image_out writes are visible
        write_output(root_x, root_y, sub_col_x, sub_col_y, local_index, size_tiles);
        workgroupBarrier(); // ensure writes complete before next work item

        work_item += stride;
    }
}

fn process_filled_root(
    root_z: u32,
    local_index: u32,
) {
    let front_z = root_z * TILE_SIZE + TILE_SIZE - 1u;

    // Update tile4_zmin for all 16 positions (first 16 threads)
    if local_index < 16u {
        atomicMax(&tile4_zmin[local_index], front_z);
    }

    // 64 threads handle 256 pixels (4 per thread): update occlusion only
    for (var k = 0u; k < 4u; k++) {
        let pixel_local = local_index * 4u + k;
        atomicMax(&voxel_zmin[pixel_local], front_z);
    }
}

// Tile4 info encoding in tile4_tape[] (Phase 1 → Phase 2 communication)
const TILE4_SKIP: u32 = 0xFFFFFFFFu;  // empty, occluded, or filled (handled in Phase 1)
// Other values: tape_start | LOCAL_TAPE_BIT — ambiguous tile needing voxel eval

fn process_ambiguous_root(
    root_x: u32, root_y: u32, root_z: u32,
    sub_col_x: u32, sub_col_y: u32,
    local_id: vec3u, local_index: u32,
    workgroup_index: u32,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let lz = local_id.z;

    let thread_tape_base = workgroup_index * 64u * TAPE_BUDGET + local_index * TAPE_BUDGET;

    let size_tiles16 = config.render_size / 16u;
    let px_base = root_x * TILE_SIZE + sub_col_x * 16u;
    let py_base = root_y * TILE_SIZE + sub_col_y * 16u;
    let tile16_x = root_x * 4u + sub_col_x;
    let tile16_y = root_y * 4u + sub_col_y;
    let tile4_local_xy = lx + ly * 4u;

    for (var z16_rev = 0u; z16_rev < 4u; z16_rev++) {
        let z16 = 3u - z16_rev;

        // Look up pre-computed tile16 status (from separate dispatch)
        let tile16_z = root_z * 4u + z16;
        let tile16_idx = tile16_x + tile16_y * size_tiles16.x
            + tile16_z * size_tiles16.x * size_tiles16.y;
        let z16_word = tile16_status[tile16_idx];
        let z16_stat = z16_word >> STATUS_SHIFT;
        let z16_tape = z16_word & TAPE_OFFSET_MASK;

        if z16_stat == ROOT_EMPTY {
            continue;
        }

        if z16_stat == ROOT_FILLED {
            let front_z_z16 = root_z * TILE_SIZE + z16 * 16u + 15u;
            atomicMax(&tile4_zmin[tile4_local_xy], front_z_z16);
            for (var vy = 0u; vy < 4u; vy++) {
                for (var vx = 0u; vx < 4u; vx++) {
                    atomicMax(&voxel_zmin[lx * 4u + vx + (ly * 4u + vy) * 16u], front_z_z16);
                }
            }
            continue;
        }

        // ================================================================
        // Phase 1: Parallel interval eval + simplification
        // Each thread evaluates its own 4^3 tile (determined by lx, ly, lz)
        // ================================================================
        let tile4_x = root_x * 16u + sub_col_x * 4u + lx;
        let tile4_y = root_y * 16u + sub_col_y * 4u + ly;
        let tile4_z = root_z * 16u + z16 * 4u + lz;
        let my_z_min_voxel = tile4_z * 4u;
        let my_z_max_voxel = my_z_min_voxel + 3u;

        var tile_result = TILE4_SKIP;

        let tile4_occ = atomicLoad(&tile4_zmin[tile4_local_xy]);
        if !(tile4_occ >= my_z_max_voxel && tile4_occ > 0u) {
            let tile4_corner = vec3u(tile4_x, tile4_y, tile4_z);
            let m = interval_inputs(tile4_corner, 4u);
            var stack = Stack();
            let out = run_tape_interval(z16_tape, m, &stack);
            let v = out.value.v;

            if v[1] < 0.0 {
                // Filled
                atomicMax(&tile4_zmin[tile4_local_xy], my_z_max_voxel);
                for (var vy = 0u; vy < 4u; vy++) {
                    for (var vx = 0u; vx < 4u; vx++) {
                        atomicMax(&voxel_zmin[lx * 4u + vx + (ly * 4u + vy) * 16u], my_z_max_voxel);
                    }
                }
            } else if !(v[0] > 0.0) {
                // Ambiguous — simplify tape for this 4^3 tile
                let local_start = simplify_tape_local(out.pos, out.count, &stack,
                                                       thread_tape_base, TAPE_BUDGET);
                if local_start != SIMPLIFY_FAILED {
                    tile_result = local_start | LOCAL_TAPE_BIT;
                } else {
                    tile_result = z16_tape;
                }
            }
        }

        tile4_tape[local_index] = tile_result;
        workgroupBarrier();

        // ================================================================
        // Phase 2: Cooperative voxel eval
        // All 64 threads work together on each ambiguous tile.
        // Thread (lx, ly, lz) evaluates one voxel per tile.
        // Process front-to-back (highest z first = tile index 63 down to 0).
        // ================================================================
        for (var t4_rev = 0u; t4_rev < 64u; t4_rev++) {
            let t4 = 63u - t4_rev;
            let info = tile4_tape[t4];

            if info == TILE4_SKIP { continue; }

            let t4_lx = t4 % 4u;
            let t4_ly = (t4 / 4u) % 4u;
            let t4_lz = t4 / 16u;
            let t4_tile4_z = root_z * 16u + z16 * 4u + t4_lz;
            let t4_z_min = t4_tile4_z * 4u;
            let tape_start = info & ~LOCAL_TAPE_BIT;
            let is_local = (info & LOCAL_TAPE_BIT) != 0u;

            let vpx_local = t4_lx * 4u + lx;
            let vpy_local = t4_ly * 4u + ly;
            let vpx = px_base + vpx_local;
            let vpy = py_base + vpy_local;
            let voxel_z = t4_z_min + (3u - lz);

            if vpx >= config.image_size.x || vpy >= config.image_size.y { continue; }

            if atomicLoad(&voxel_zmin[vpx_local + vpy_local * 16u]) >= voxel_z {
                continue;
            }

            let fm = float_transformed_inputs(f32(vpx), f32(vpy), f32(voxel_z));
            var fval: FloatValue;
            if is_local {
                fval = run_tape_float(tape_start, fm);
            } else {
                fval = run_tape_float_root(tape_start, fm);
            }

            if fval.v < 0.0 {
                atomicMax(&voxel_zmin[vpx_local + vpy_local * 16u], voxel_z);
            }
        }

        // Propagate voxel hits to coarse tile4 occlusion.
        // tile4_zmin[xy] = min(voxel_zmin[px]) across all 16 pixels in the 4x4 column.
        // Only update when ALL 16 pixels have hits (conservative).
        workgroupBarrier();  // ensure all Phase 2 voxel_zmin writes are visible

        if lz == 0u {
            var min_z = 0xFFFFFFFFu;
            for (var vy = 0u; vy < 4u; vy++) {
                for (var vx = 0u; vx < 4u; vx++) {
                    min_z = min(min_z, atomicLoad(&voxel_zmin[lx * 4u + vx + (ly * 4u + vy) * 16u]));
                }
            }
            if min_z > 0u {
                atomicMax(&tile4_zmin[lx + ly * 4u], min_z);
            }
        }

        workgroupBarrier();  // ensure tile4_zmin visible for next z16 Phase 1
    }
}

fn write_output(
    root_x: u32, root_y: u32,
    sub_col_x: u32, sub_col_y: u32,
    local_index: u32,
    size_tiles: vec3u,
) {
    let px_base = root_x * TILE_SIZE + sub_col_x * 16u;
    let py_base = root_y * TILE_SIZE + sub_col_y * 16u;

    for (var k = 0u; k < 4u; k++) {
        let pixel_local = local_index * 4u + k;
        let px = px_base + (pixel_local % 16u);
        let py = py_base + (pixel_local / 16u);
        if px >= config.image_size.x || py >= config.image_size.y { continue; }

        let z = atomicLoad(&voxel_zmin[pixel_local]);
        if z == 0u { continue; }

        let pixel_idx = px + py * config.image_size.x;

        // Compute gradient using tile16 tape
        let size16 = config.render_size / 16u;
        let t16_x = px / 16u;
        let t16_y = py / 16u;
        let t16_z = z / 16u;
        let t16_idx = t16_x + t16_y * size16.x + t16_z * size16.x * size16.y;
        let t16_word = tile16_status[t16_idx];
        let tape_start = t16_word & TAPE_OFFSET_MASK;

        let gm = grad_transformed_inputs(f32(px), f32(py), f32(z));
        let grad = run_tape_grad_root(tape_start, gm);
        let normal = normalize(grad.v.xyz);
        image_out[pixel_idx] = vec4f(f32(z), normal.x, normal.y, normal.z);
    }
}
