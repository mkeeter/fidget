// Interval evaluation stage for raymarching shader
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

/// Per-state IO bindings
@group(1) @binding(0) var<storage, read> tiles_in: TileListInput;
@group(1) @binding(1) var<storage, read> tile_zmin: array<u32>;

@group(1) @binding(2) var<storage, read_write> subtiles_out: TileListOutput;
@group(1) @binding(3) var<storage, read_write> subtile_zmin: array<atomic<u32>>;

/// Count to clear (unused in this pass)
@group(1) @binding(4) var<storage, read_write> count_clear: array<atomic<u32>, 4>;

/// Input tile size; one input tile maps to a 4x4x4 workgroup
override TILE_SIZE: u32;

/// Output tile size, must be TILE_SIZE / 4; one output tile maps to one thread
override SUBTILE_SIZE: u32;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Reset an unused counter
    atomicStore(&count_clear[local_id.x], 0u);

    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile_index = workgroup_id.x + workgroup_id.y * 32768;
    if (active_tile_index >= tiles_in.count) {
        return;
    }

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
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

    // Compute transformed interval regions
    let ix = vec2f(f32(corner_pos.x), f32(corner_pos.x + SUBTILE_SIZE));
    let iy = vec2f(f32(corner_pos.y), f32(corner_pos.y + SUBTILE_SIZE));
    let iz = vec2f(f32(corner_pos.z), f32(corner_pos.z + SUBTILE_SIZE));
    var ts = array(Value(), Value(), Value(), Value());
    for (var i = 0; i < 4; i++) {
        ts[i] = op_add(
            op_add(
                op_mul(build_imm(config.mat[0][i]), Value(ix)),
                op_mul(build_imm(config.mat[1][i]), Value(iy)),
            ),
            op_add(
                op_mul(build_imm(config.mat[2][i]), Value(iz)),
                build_imm(config.mat[3][i]),
            ),
        );
    }

    // Build up input map
    var m = array(Value(), Value(), Value());
    if (config.axes.x < 3) {
        m[config.axes.x] = op_div(ts[0], ts[3]);
    }
    if (config.axes.y < 3) {
        m[config.axes.y] = op_div(ts[1], ts[3]);
    }
    if (config.axes.z < 3) {
        m[config.axes.z] = op_div(ts[2], ts[3]);
    }

    // Last-minute check to see if anyone filled out this tile
    if (atomicLoad(&subtile_zmin[subtile_index_xy]) >= corner_pos.z + SUBTILE_SIZE) {
        return;
    }

    // Do the actual interpreter work
    let out = run_tape(0u, m)[0];

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

struct Value {
    v: vec2f,
}

fn has_nan(i: Value) -> bool {
    return i.v.x != i.v.x || i.v.y != i.v.y;
}

fn nan_i() -> Value {
    return Value(vec2f(nan_f32()));
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v.yx);
}

fn op_abs(lhs: Value) -> Value {
    if lhs.v[0] < 0.0 {
        if lhs.v[1] > 0.0 {
            return Value(vec2f(0.0, max(lhs.v[1], -lhs.v[0])));
        } else {
            return Value(vec2f(-lhs.v[1], -lhs.v[0]));
        }
    } else {
        return lhs;
    }
}

fn op_recip(lhs: Value) -> Value {
    if (lhs.v[0] > 0.0 || lhs.v[1] < 0.0) {
        return Value(vec2f(1.0 / lhs.v[1], 1.0 / lhs.v[0]));
    } else {
        return nan_i();
    }
}

fn op_sqrt(lhs: Value) -> Value {
    if (lhs.v[0] >= 0.0) {
        return Value(sqrt(lhs.v));
    } else {
        return nan_i();
    }
}

fn op_floor(lhs: Value) -> Value {
    return Value(floor(lhs.v));
}

fn op_ceil(lhs: Value) -> Value {
    return Value(ceil(lhs.v));
}

fn op_round(lhs: Value) -> Value {
    return Value(round(lhs.v));
}

fn op_square(lhs: Value) -> Value {
    if (lhs.v[1] < 0.0) {
        return Value(vec2f(lhs.v[1] * lhs.v[1], lhs.v[0] * lhs.v[0]));
    } else if (lhs.v[0] > 0.0) {
        return Value(vec2f(lhs.v[0] * lhs.v[0], lhs.v[1] * lhs.v[1]));
    } else if (has_nan(lhs)) {
        return nan_i();
    } else {
        let v = max(abs(lhs.v[0]), abs(lhs.v[1]));
        return Value(vec2f(0.0, v * v));
    }
}

fn op_sin(lhs: Value) -> Value {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_cos(lhs: Value) -> Value {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_tan(lhs: Value) -> Value {
    let size = lhs.v[1] - lhs.v[0];
    if (size >= 3.14159265f) {
        return nan_i();
    } else {
        let lower = tan(lhs.v[0]);
        let upper = tan(lhs.v[1]);
        if (upper >= lower) {
            return Value(vec2f(lower, upper));
        } else {
            return nan_i();
        }
    }
}

fn op_asin(lhs: Value) -> Value {
    if (lhs.v[0] < -1.0 || lhs.v[1] > 1.0) {
        return nan_i();
    } else {
        return Value(asin(lhs.v));
    }
}

fn op_acos(lhs: Value) -> Value {
    if (lhs.v[0] < -1.0 || lhs.v[1] > 1.0) {
        return nan_i();
    } else {
        return Value(acos(lhs.v).yx);
    }
}

fn op_atan(lhs: Value) -> Value {
    return Value(atan(lhs.v));
}

fn op_exp(lhs: Value) -> Value {
    return Value(exp(lhs.v));
}

fn op_log(lhs: Value) -> Value {
    if (lhs.v[0] < 0.0) {
        return nan_i();
    } else {
        return Value(log(lhs.v));
    }
}

fn op_not(lhs: Value) -> Value {
    if (!contains_i(lhs, 0.0) && !has_nan(lhs)) {
        return Value(vec2f(0.0));
    } else if (lhs.v[0] == 0.0 && lhs.v[1] == 0.0) {
        return Value(vec2f(1.0));
    } else {
        return Value(vec2f(0.0, 1.0));
    }
}

fn contains_i(i: Value, v: f32) -> bool {
    return (i.v[0] <= v && v <= i.v[1]);
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (lhs.v[1] < rhs.v[0]) {
        return Value(vec2f(-1.0));
    } else if (lhs.v[0] > rhs.v[1]) {
        return Value(vec2f(1.0));
    } else if (lhs.v[0] == lhs.v[1] && rhs.v[0] == rhs.v[1] && lhs.v[0] == rhs.v[0]) {
        return Value(vec2f(0.0));
    } else {
        return Value(vec2f(-1.0, 1.0));
    }
}

fn op_and(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (lhs.v[0] == 0.0 && lhs.v[1] == 0.0) {
        return Value(vec2f(0.0));
    } else if (!contains_i(lhs, 0.0)) {
        return rhs;
    } else {
        return Value(vec2f(min(rhs.v[0], 0.0), max(rhs.v[1], 0.0)));
    }
}

fn op_or(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (!contains_i(lhs, 0.0)) {
        return lhs;
    } else if (lhs.v[0] == 0.0 && lhs.v[1] == 0.0) {
        return rhs;
    } else {
        return Value(vec2f(min(lhs.v[0], rhs.v[0]), max(lhs.v[0], rhs.v[0])));
    }
}

fn build_imm(v: f32) -> Value {
    return Value(vec2f(v));
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if (r < 0.0) {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn op_mod(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    } else if (rhs.v[0] == rhs.v[1] && rhs.v[0] > 0.0) {
        let a = lhs.v[0] / rhs.v[0];
        let b = lhs.v[1] / rhs.v[0];
        if (a != floor(a) && floor(a) == floor(b)) {
            return Value(vec2f(
                rem_euclid(lhs.v[0], rhs.v[0]),
                rem_euclid(lhs.v[1], rhs.v[0]),
            ));
        } else {
            return Value(vec2f(0.0, abs(rhs.v[0])));
        }
    } else {
        return Value(vec2f(0.0, max(abs(rhs.v[0]), abs(rhs.v[1]))));
    }
}

fn op_add(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v + rhs.v);
}

fn op_sub(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v - rhs.v.yx);
}

fn op_min(lhs: Value, rhs: Value) -> Value {
    return Value(min(lhs.v, rhs.v));
}

fn op_max(lhs: Value, rhs: Value) -> Value {
    return Value(max(lhs.v, rhs.v));
}

fn op_mul(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    }
    let ab = lhs.v * rhs.v;
    let cd = lhs.v.yx * rhs.v;
    return Value(vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    ));
}

fn op_div(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || contains_i(rhs, 0.0)) {
        return nan_i();
    }
    let ab = lhs.v / rhs.v;
    let cd = lhs.v.yx / rhs.v;
    return Value(vec2f(
        min(min(ab[0], ab[1]), min(cd[0], cd[1])),
        max(max(ab[0], ab[1]), max(cd[0], cd[1])),
    ));
}

fn op_atan2(lhs: Value, rhs: Value) -> Value {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    }
    return Value(vec2f(-3.141592654, 3.141592654));
}
