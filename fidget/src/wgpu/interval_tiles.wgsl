// Interval evaluation stage for raymarching shader
//
// This shader must also be concatenated with `interpreter_i.wgsl`, which
// provides the `run_tape` function.

/// Render configuration and tape data
@group(0) @binding(0) var<storage> config: Config;

/// Input tape(s), serialized to bytecode
///
/// This array contains many tapes concatenated together; tape start positions
/// are recorded in the `tile64_tapes` input array.
@group(0) @binding(1) var<storage, read> tile64_tapes: array<u32>;

@group(0) @binding(2) var<storage, read> tiles_in: TileListInput;
@group(0) @binding(3) var<storage, read> tile_zmin: array<u32>;

@group(0) @binding(4) var<storage, read_write> subtiles_out: TileListOutput;
@group(0) @binding(5) var<storage, read_write> subtile_zmin: array<atomic<u32>>;

/// Count to clear (unused in this pass)
@group(0) @binding(6) var<storage, read_write> count_clear: array<atomic<u32>, 4>;

override TILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    let SUBTILE_SIZE = TILE_SIZE / 4u;

    // Reset an unused counter
    atomicStore(&count_clear[local_id.x], 0u);

    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile_index = workgroup_id.x + workgroup_id.y * 32768;
    if (active_tile_index >= tiles_in.count) {
        return;
    }

    // Convert to a size in tile units
    let size64 = (config.image_size + 63u) / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Get global tile position, in tile coordinates
    let t = tiles_in.active_tiles[active_tile_index];
    let tx = t % size_tiles.x;
    let ty = (t / size_tiles.x) % size_tiles.y;
    let tz = (t / (size_tiles.x * size_tiles.y)) % size_tiles.z;
    let tile_corner = vec3u(tx, ty, tz);

    // Subtile corner position
    let subtile_corner = tile_corner * 4 + local_id;
    let subtile_index_xy = subtile_corner.x + subtile_corner.y * size_subtiles.x;

    // Subtile corner position, in voxels
    let corner_pos = subtile_corner * SUBTILE_SIZE;

    // Check for Z masking from tile
    let tile_index_xy = tile_corner.x + tile_corner.y * size_tiles.x;
    if (tile_zmin[tile_index_xy] >= corner_pos.z) {
        atomicMax(&subtile_zmin[subtile_index_xy], tile_zmin[tile_index_xy]);
        return;
    }

    // Pick out our starting tape, based on absolute position
    let tile64_pos = corner_pos / 64;
    let tile64_index = tile64_pos.x +
        tile64_pos.y * size64.x +
        tile64_pos.z * size64.x * size64.y;
    let tape_start = tile64_tapes[tile64_index];

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + SUBTILE_SIZE));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + SUBTILE_SIZE));
    let iz = vec2f(f32(corner_pos.z), f32(corner_pos.z + SUBTILE_SIZE));
    var ts = mat4x2f(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
    for (var i = 0; i < 4; i++) {
        ts[i] = mul_i(vec2f(config.mat[0][i]), ix)
            + mul_i(vec2f(config.mat[1][i]), iy)
            + mul_i(vec2f(config.mat[2][i]), iz)
            + vec2f(config.mat[3][i]);
    }

    // Build up input map
    var m = mat4x2f(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
    if (config.axes.x < 4) {
        m[config.axes.x] = div_i(ts[0], ts[3]);
    }
    if (config.axes.y < 4) {
        m[config.axes.y] = div_i(ts[1], ts[3]);
    }
    if (config.axes.z < 4) {
        m[config.axes.z] = div_i(ts[2], ts[3]);
    }

    // Last-minute check to see if anyone filled out this tile
    if (atomicLoad(&subtile_zmin[subtile_index_xy]) >= corner_pos.z + SUBTILE_SIZE) {
        return;
    }

    // Do the actual interpreter work
    let out = run_tape_i(tape_start, m);

    if (out[1] < 0.0) {
        // Full, write to subtile_zmin
        atomicMax(&subtile_zmin[subtile_index_xy], corner_pos.z + SUBTILE_SIZE);
    } else if (out[0] > 0.0) {
        // Empty, nothing to do here
    } else {
        let offset = atomicAdd(&subtiles_out.count, 1u);
        let subtile_index_xyz = subtile_corner.x +
            (subtile_corner.y * size_subtiles.x) +
            (subtile_corner.z * size_subtiles.x * size_subtiles.y);
        subtiles_out.active_tiles[offset] = subtile_index_xyz;

        let count = offset + 1u;
        let wg_dispatch_x = min(count, 32768u);
        let wg_dispatch_y = (count + 32767u) / 32768u;
        atomicMax(&subtiles_out.wg_size[0], wg_dispatch_x);
        atomicMax(&subtiles_out.wg_size[1], wg_dispatch_y);
        atomicMax(&subtiles_out.wg_size[2], 1u);
    }
}

fn has_nan(i: vec2f) -> bool {
    return i.x != i.x || i.y != i.y;
}

fn nan_i() -> vec2f {
    let f = bitcast<f32>(0x7FC00000);
    return vec2f(f, f);
}

fn neg_i(lhs: vec2f) -> vec2f {
    return -lhs.yx;
}

fn abs_i(lhs: vec2f) -> vec2f {
    if lhs[0] < 0.0 {
        if lhs[1] > 0.0 {
            return vec2f(0.0, max(lhs[1], -lhs[0]));
        } else {
            return vec2f(-lhs[1], -lhs[0]);
        }
    } else {
        return lhs;
    }
}

fn recip_i(lhs: vec2f) -> vec2f {
    if (lhs[0] > 0.0 || lhs[1] < 0.0) {
        return vec2f(1.0 / lhs[1], 1.0 / lhs[0]);
    } else {
        return nan_i();
    }
}

fn sqrt_i(lhs: vec2f) -> vec2f {
    if (lhs[0] >= 0.0) {
        return sqrt(lhs);
    } else {
        return nan_i();
    }
}

fn square_i(lhs: vec2f) -> vec2f {
    if (lhs[1] < 0.0) {
        return vec2f(lhs[1] * lhs[1], lhs[0] * lhs[0]);
    } else if (lhs[0] > 0.0) {
        return vec2f(lhs[0] * lhs[0], lhs[1] * lhs[1]);
    } else if (has_nan(lhs)) {
        return nan_i();
    } else {
        let v = max(abs(lhs[0]), abs(lhs[1]));
        return vec2f(0.0, v * v);
    }
}

fn sin_i(lhs: vec2f) -> vec2f {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return vec2f(-1.0, 1.0);
    }
}

fn cos_i(lhs: vec2f) -> vec2f {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return vec2f(-1.0, 1.0);
    }
}

fn tan_i(lhs: vec2f) -> vec2f {
    let size = lhs[1] - lhs[0];
    if (size >= 3.14159265f) {
        return nan_i();
    } else {
        let lower = tan(lhs[0]);
        let upper = tan(lhs[1]);
        if (upper >= lower) {
            return vec2f(lower, upper);
        } else {
            return nan_i();
        }
    }
}

fn asin_i(lhs: vec2f) -> vec2f {
    if (lhs[0] < -1.0 || lhs[1] > 1.0) {
        return nan_i();
    } else {
        return asin(lhs);
    }
}

fn acos_i(lhs: vec2f) -> vec2f {
    if (lhs[0] < -1.0 || lhs[1] > 1.0) {
        return nan_i();
    } else {
        return acos(lhs).yx;
    }
}

fn atan_i(lhs: vec2f) -> vec2f {
    return atan(lhs);
}

fn exp_i(lhs: vec2f) -> vec2f {
    return exp(lhs);
}

fn log_i(lhs: vec2f) -> vec2f {
    if (lhs[0] < 0.0) {
        return nan_i();
    } else {
        return log(lhs);
    }
}

fn not_i(lhs: vec2f) -> vec2f {
    if (!contains_i(lhs, 0.0) && !has_nan(lhs)) {
        return vec2f(0.0);
    } else if (lhs[0] == 0.0 && lhs[1] == 0.0) {
        return vec2f(1.0);
    } else {
        return vec2f(0.0, 1.0);
    }
}

fn contains_i(i: vec2f, v: f32) -> bool {
    return (i[0] <= v && v <= i[1]);
}

fn compare_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (lhs[1] < rhs[0]) {
        return vec2f(-1.0);
    } else if (lhs[0] > rhs[1]) {
        return vec2f(1.0);
    } else if (lhs[0] == lhs[1] && rhs[0] == rhs[1] && lhs[0] == rhs[0]) {
        return vec2f(0.0);
    } else {
        return vec2f(-1.0, 1.0);
    }
}

fn and_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (lhs[0] == 0.0 && lhs[1] == 0.0) {
        return vec2f(0.0);
    } else if (!contains_i(lhs, 0.0)) {
        return rhs;
    } else {
        return vec2f(min(rhs[0], 0.0), max(rhs[1], 0.0));
    }
}

fn or_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (!contains_i(lhs, 0.0)) {
        return lhs;
    } else if (lhs[0] == 0.0 && lhs[1] == 0.0) {
        return rhs;
    } else {
        return vec2f(min(lhs[0], rhs[0]), max(lhs[0], rhs[0]));
    }
}

fn read_imm_i(i: ptr<function, u32>) -> vec2f {
    let imm = read_imm_f(i);
    return vec2f(imm);
}

fn read_imm_f(i: ptr<function, u32>) -> f32 {
    let imm = bitcast<f32>(config.tape_data[*i]);
    *i = *i + 1;
    return imm;
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if (r < 0.0) {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn mod_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (rhs[0] == rhs[1] && rhs[0] > 0.0) {
        let a = lhs[0] / rhs[0];
        let b = lhs[1] / rhs[0];
        if (a != floor(a) && floor(a) == floor(b)) {
            return vec2f(
                rem_euclid(lhs[0], rhs[0]),
                rem_euclid(lhs[1], rhs[0]),
            );
        } else {
            return vec2f(0.0, max(abs(rhs[0]), abs(rhs[1])));
        }
    } else {
        return vec2f(0.0, max(abs(rhs[0]), abs(rhs[1])));
    }
}

fn mul_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    }
    let ab = lhs * rhs;
    let cd = lhs.yx * rhs;
    return vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    );
}

fn div_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || contains_i(rhs, 0.0)) {
        return nan_i();
    }
    let ab = lhs / rhs;
    let cd = lhs.yx / rhs;
    return vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    );
}

fn atan2_i(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    }
    return vec2f(-3.141592654, 3.141592654);
}

fn run_tape_i(start: u32, inputs: mat4x2f) -> vec2f {
    var i: u32 = start;
    var reg: array<vec2f, 256>;
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
            case OP_COPY_IMM:    { reg[op[1]] = read_imm_i(&i); }
            case OP_NEG_REG:     { reg[op[1]] = neg_i(reg[op[2]]); }
            case OP_ABS_REG:     { reg[op[1]] = abs_i(reg[op[2]]); }
            case OP_RECIP_REG:   { reg[op[1]] = recip_i(reg[op[2]]); }
            case OP_SQRT_REG:    { reg[op[1]] = sqrt_i(reg[op[2]]); }
            case OP_SQUARE_REG:  { reg[op[1]] = square_i(reg[op[2]]); }
            case OP_FLOOR_REG:   { reg[op[1]] = floor(reg[op[2]]); }
            case OP_CEIL_REG:    { reg[op[1]] = ceil(reg[op[2]]); }
            case OP_ROUND_REG:   { reg[op[1]] = round(reg[op[2]]); }
            case OP_SIN_REG:     { reg[op[1]] = sin_i(reg[op[2]]); }
            case OP_COS_REG:     { reg[op[1]] = cos_i(reg[op[2]]); }
            case OP_TAN_REG:     { reg[op[1]] = tan_i(reg[op[2]]); }
            case OP_ASIN_REG:    { reg[op[1]] = asin_i(reg[op[2]]); }
            case OP_ACOS_REG:    { reg[op[1]] = acos_i(reg[op[2]]); }
            case OP_ATAN_REG:    { reg[op[1]] = atan_i(reg[op[2]]); }
            case OP_EXP_REG:     { reg[op[1]] = exp_i(reg[op[2]]); }
            case OP_LN_REG:      { reg[op[1]] = log_i(reg[op[2]]); }
            case OP_NOT_REG:     { reg[op[1]] = not_i(reg[op[2]]); }
            case OP_ADD_REG_IMM:  { reg[op[1]] = reg[op[2]] + read_imm_i(&i); }
            case OP_MUL_REG_IMM:  { reg[op[1]] = mul_i(reg[op[2]], read_imm_i(&i)); }
            case OP_DIV_REG_IMM:  { reg[op[1]] = div_i(reg[op[2]], read_imm_i(&i)); }
            case OP_SUB_REG_IMM:  { reg[op[1]] = reg[op[2]] - read_imm_i(&i); }
            case OP_MOD_REG_IMM:  { reg[op[1]] = mod_i(reg[op[2]], read_imm_i(&i)); }
            case OP_ATAN_REG_IMM: { reg[op[1]] = atan2_i(reg[op[2]], read_imm_i(&i)); }
            case OP_COMPARE_REG_IMM:  { reg[op[1]] = compare_i(reg[op[2]], read_imm_i(&i)); }

            case OP_DIV_IMM_REG:      { reg[op[1]] = div_i(read_imm_i(&i), reg[op[2]]); }
            case OP_SUB_IMM_REG:      { reg[op[1]] = read_imm_i(&i) - reg[op[2]].yx; }
            case OP_MOD_IMM_REG:      { reg[op[1]] = mod_i(read_imm_i(&i), reg[op[2]]); }
            case OP_ATAN_IMM_REG:     { reg[op[1]] = atan2_i(read_imm_i(&i), reg[op[2]]); }
            case OP_COMPARE_IMM_REG:  { reg[op[1]] = compare_i(read_imm_i(&i), reg[op[2]]); }

            case OP_MIN_REG_IMM:  { reg[op[1]] = min(reg[op[2]], read_imm_i(&i)); }
            case OP_MAX_REG_IMM:  { reg[op[1]] = max(reg[op[2]], read_imm_i(&i)); }
            case OP_AND_REG_IMM:  { reg[op[1]] = and_i(reg[op[2]], read_imm_i(&i)); }
            case OP_OR_REG_IMM:   { reg[op[1]] = or_i(reg[op[2]], read_imm_i(&i)); }

            case OP_ADD_REG_REG:      { reg[op[1]] = reg[op[2]] + reg[op[3]]; }
            case OP_MUL_REG_REG:      { reg[op[1]] = mul_i(reg[op[2]], reg[op[3]]); }
            case OP_DIV_REG_REG:      { reg[op[1]] = div_i(reg[op[2]], reg[op[3]]); }
            case OP_SUB_REG_REG:      { reg[op[1]] = reg[op[2]] - reg[op[3]].yx; }
            case OP_COMPARE_REG_REG:  { reg[op[1]] = compare_i(reg[op[2]], reg[op[3]]); }
            case OP_ATAN_REG_REG:      { reg[op[1]] = atan2_i(reg[op[2]], reg[op[3]]); }
            case OP_MOD_REG_REG:      { reg[op[1]] = mod_i(reg[op[2]], reg[op[3]]); }

            case OP_MIN_REG_REG:      { reg[op[1]] = min(reg[op[2]], reg[op[3]]); }
            case OP_MAX_REG_REG:      { reg[op[1]] = max(reg[op[2]], reg[op[3]]); }
            case OP_AND_REG_REG:      { reg[op[1]] = and_i(reg[op[2]], reg[op[3]]); }
            case OP_OR_REG_REG:       { reg[op[1]] = or_i(reg[op[2]], reg[op[3]]); }

            case OP_LOAD, OP_STORE: {
                // Not implemented!
                break;
            }
            default: {
                break;
            }
        }
    }
    return nan_i(); // unknown opcode
}
