// VM interpreter for floating-point values, using voxel tiles

/// Render configuration and tape data
@group(0) @binding(0) var<storage> config: Config;

/// Tile data (64^3, dense)
@group(0) @binding(1) var<storage, read> tile64_tapes: array<u32>;

@group(0) @binding(2) var<storage, read> tiles_in: TileListInput;
@group(0) @binding(3) var<storage, read> tile4_zmin: array<u32>;

/// Output array, image size
@group(0) @binding(4) var<storage, read_write> result: array<atomic<u32>>;

/// Count to clear (unused in this pass)
@group(0) @binding(5) var<storage, read_write> count_clear: array<atomic<u32>, 4>;

var<workgroup> skip_count: atomic<u32>;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Reset an unused counter
    atomicStore(&count_clear[local_id.x], 0u);

    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile4_index = workgroup_id.x + workgroup_id.y * 32768;
    if (active_tile4_index >= tiles_in.count) {
        return;
    }

    // Convert to a size in tile units
    let size64 = (config.image_size + 63u) / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;

    // Get global tile position, in tile4 coordinates
    let t = tiles_in.active_tiles[active_tile4_index];
    let tx = t % size4.x;
    let ty = (t / size4.x) % size4.y;
    let tz = (t / (size4.x * size4.y)) % size4.z;
    let tile4_corner = vec3u(tx, ty, tz);

    // Subtile corner position, in voxels
    let corner_pos = tile4_corner * 4 + local_id;

    let tile4_index_xy = tx + ty * size4.x;
    let pixel_index_xy = corner_pos.x + corner_pos.y * config.image_size.x;
    var already_skipped = false;
    if (tile4_zmin[tile4_index_xy] >= corner_pos.z) {
        atomicMax(&result[pixel_index_xy], tile4_zmin[tile4_index_xy]);
        atomicAdd(&skip_count, 1u);
        already_skipped = true;
    }

    // Pick out our starting tape, based on absolute position
    let tile64_pos = corner_pos / 64;
    let tile64_index = tile64_pos.x +
        tile64_pos.y * size64.x +
        tile64_pos.z * size64.x * size64.y;
    let tape_start = tile64_tapes[tile64_index];

    // Last chance to bail out
    if (!already_skipped && atomicLoad(&result[pixel_index_xy]) >= u32(corner_pos.z)) {
        atomicAdd(&skip_count, 1u);
    }
    workgroupBarrier();
    if (atomicLoad(&skip_count) == 64u) {
        return;
    }

    // Voxel evaluation: start by building up the input map
    var m = vec4f(0.0);

    let pos_pixels = vec4f(vec3f(corner_pos), 1.0);
    let pos_model = config.mat * pos_pixels;

    if (config.axes.x < 4) {
        m[config.axes.x] = pos_model.x / pos_model.w;
    }
    if (config.axes.y < 4) {
        m[config.axes.y] = pos_model.y / pos_model.w;
    }
    if (config.axes.z < 4) {
        m[config.axes.z] = pos_model.z / pos_model.w;
    }

    // Do the actual interpreter work
    let out = run_tape(tape_start, m, local_id.x + local_id.y * 4 + local_id.z * 16);

    if (out < 0.0) {
        atomicMax(&result[pixel_index_xy], corner_pos.z);
    }
}

fn nan_f32() -> f32 {
    return bitcast<f32>(0x7FC00000);
}

var<workgroup> tape_scratch: array<u32, 64>;

fn compare_f32(lhs: f32, rhs: f32) -> f32 {
    if (lhs < rhs) {
        return -1.0;
    } else if (lhs > rhs) {
        return 1.0;
    } else if (lhs == rhs) {
        return 0.0;
    } else {
        return nan_f32();
    }
}

fn and_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs == 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn or_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs != 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn not_f32(lhs: f32) -> f32 {
    return f32(lhs != 0.0);
}

fn read_imm_f32(i: ptr<function, u32>) -> f32 {
    let imm = bitcast<f32>(tape_scratch[*i]);
    *i = *i + 1;
    return imm;
}

fn mod_f32(lhs: f32, rhs: f32) -> f32 {
    var out = lhs % rhs;
    out -= rhs * min(0.0, floor(out / rhs));
    return out;
}

fn run_tape(start: u32, inputs: vec4f, wg_offset: u32) -> f32 {
    var i: u32 = start;
    var reg: array<f32, 256>;
    while (true) {
        let d = config.tape_data[i + wg_offset];
        workgroupBarrier();
        tape_scratch[wg_offset] = d;
        workgroupBarrier();

        var j: u32 = 0;
        while (j < 63) {
            let op = unpack4xU8(tape_scratch[j]);
            j = j + 1;
            switch op[0] {
                case OP_OUTPUT: {
                    // XXX we're not actually writing to an output slot here
                    let imm = tape_scratch[j];
                    j = j + 1;
                    return reg[op[1]];
                }
                case OP_INPUT: {
                    let imm = tape_scratch[j];
                    j = j + 1;
                    reg[op[1]] = inputs[imm];
                }
                case OP_COPY_REG:    { reg[op[1]] = reg[op[2]]; }
                case OP_COPY_IMM:    { reg[op[1]] = read_imm_f32(&j); }
                case OP_NEG_REG:     { reg[op[1]] = -reg[op[2]]; }
                case OP_ABS_REG:     { reg[op[1]] = abs(reg[op[2]]); }
                case OP_RECIP_REG:   { reg[op[1]] = 1.0 / reg[op[2]]; }
                case OP_SQRT_REG:    { reg[op[1]] = sqrt(reg[op[2]]); }
                case OP_SQUARE_REG: {
                    let v = reg[op[2]];
                    reg[op[1]] = v * v;
                }
                case OP_FLOOR_REG:   { reg[op[1]] = floor(reg[op[2]]); }
                case OP_CEIL_REG:    { reg[op[1]] = ceil(reg[op[2]]); }
                case OP_ROUND_REG:   { reg[op[1]] = round(reg[op[2]]); }
                case OP_SIN_REG:     { reg[op[1]] = sin(reg[op[2]]); }
                case OP_COS_REG:     { reg[op[1]] = cos(reg[op[2]]); }
                case OP_TAN_REG:     { reg[op[1]] = tan(reg[op[2]]); }
                case OP_ASIN_REG:    { reg[op[1]] = asin(reg[op[2]]); }
                case OP_ACOS_REG:    { reg[op[1]] = acos(reg[op[2]]); }
                case OP_ATAN_REG:    { reg[op[1]] = atan(reg[op[2]]); }
                case OP_EXP_REG:     { reg[op[1]] = exp(reg[op[2]]); }
                case OP_LN_REG:      { reg[op[1]] = log(reg[op[2]]); }
                case OP_NOT_REG:     { reg[op[1]] = not_f32(reg[op[2]]); }
                case OP_ADD_REG_IMM:  { reg[op[1]] = reg[op[2]] + read_imm_f32(&j); }
                case OP_MUL_REG_IMM:  { reg[op[1]] = reg[op[2]] * read_imm_f32(&j); }
                case OP_DIV_REG_IMM:  { reg[op[1]] = reg[op[2]] / read_imm_f32(&j); }
                case OP_SUB_REG_IMM:  { reg[op[1]] = reg[op[2]] - read_imm_f32(&j); }
                case OP_MOD_REG_IMM:  { reg[op[1]] = mod_f32(reg[op[2]], read_imm_f32(&j)); }
                case OP_ATAN_REG_IMM: { reg[op[1]] = atan2(reg[op[2]], read_imm_f32(&j)); }
                case OP_COMPARE_REG_IMM:  { reg[op[1]] = compare_f32(reg[op[2]], read_imm_f32(&j)); }

                case OP_DIV_IMM_REG:      { reg[op[1]] = read_imm_f32(&j) / reg[op[2]]; }
                case OP_SUB_IMM_REG:      { reg[op[1]] = read_imm_f32(&j) - reg[op[2]]; }
                case OP_MOD_IMM_REG:      { reg[op[1]] = mod_f32(read_imm_f32(&j), reg[op[2]]); }
                case OP_ATAN_IMM_REG:     { reg[op[1]] = atan2(read_imm_f32(&j), reg[op[2]]); }
                case OP_COMPARE_IMM_REG:  { reg[op[1]] = compare_f32(read_imm_f32(&j), reg[op[2]]); }

                case OP_MIN_REG_IMM:  { reg[op[1]] = min(reg[op[2]], read_imm_f32(&j)); }
                case OP_MAX_REG_IMM:  { reg[op[1]] = max(reg[op[2]], read_imm_f32(&j)); }
                case OP_AND_REG_IMM:  { reg[op[1]] = and_f32(reg[op[2]], read_imm_f32(&j)); }
                case OP_OR_REG_IMM:   { reg[op[1]] = or_f32(reg[op[2]], read_imm_f32(&j)); }

                case OP_ADD_REG_REG:      { reg[op[1]] = reg[op[2]] + reg[op[3]]; }
                case OP_MUL_REG_REG:      { reg[op[1]] = reg[op[2]] * reg[op[3]]; }
                case OP_DIV_REG_REG:      { reg[op[1]] = reg[op[2]] / reg[op[3]]; }
                case OP_SUB_REG_REG:      { reg[op[1]] = reg[op[2]] - reg[op[3]]; }
                case OP_COMPARE_REG_REG:  { reg[op[1]] = compare_f32(reg[op[2]], reg[op[3]]); }
                case OP_ATAN_REG_REG:      { reg[op[1]] = atan2(reg[op[2]], reg[op[3]]); }
                case OP_MOD_REG_REG:      { reg[op[1]] = mod_f32(reg[op[2]], reg[op[3]]); }

                case OP_MIN_REG_REG:      { reg[op[1]] = min(reg[op[2]], reg[op[3]]); }
                case OP_MAX_REG_REG:      { reg[op[1]] = max(reg[op[2]], reg[op[3]]); }
                case OP_AND_REG_REG:      { reg[op[1]] = and_f32(reg[op[2]], reg[op[3]]); }
                case OP_OR_REG_REG:       { reg[op[1]] = or_f32(reg[op[2]], reg[op[3]]); }

                case OP_LOAD, OP_STORE: {
                    // Not implemented!
                    return nan_f32();
                }
                default: {
                    return nan_f32();
                }
            }
        }
        i += j;
    }
    return nan_f32(); // unknown opcode
}

