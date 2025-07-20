// Interval evaluation stage for raymarching shader

/// Render configuration and tape data
@group(0) @binding(0) var<uniform> config: Config;

@group(0) @binding(1) var<storage, read> tiles_in: TileListInput;
@group(0) @binding(2) var<storage, read> tile_zmin: array<u32>;

@group(0) @binding(3) var<storage, read_write> subtiles_out: TileListOutput;
@group(0) @binding(4) var<storage, read_write> subtile_zmin: array<atomic<u32>>;

/// Count to clear (unused in this pass)
@group(0) @binding(5) var<storage, read_write> count_clear: array<atomic<u32>, 4>;

@compute @workgroup_size(4, 4, 4)
fn interval_tile_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    let TILE_SIZE = tiles_in.size;
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
    var ts = mat4x2f(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0));
    for (var i = 0; i < 4; i++) {
        ts[i] = op_mul(build_imm(config.mat[0][i]), ix)
            + op_mul(build_imm(config.mat[1][i]), iy)
            + op_mul(build_imm(config.mat[2][i]), iz)
            + build_imm(config.mat[3][i]);
    }

    // Build up input map
    var m = array(vec2f(0.0), vec2f(0.0), vec2f(0.0));
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
    let out = run_tape(m)[0];

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
    return vec2f(nan_f32());
}

fn op_neg(lhs: vec2f) -> vec2f {
    return -lhs.yx;
}

fn op_abs(lhs: vec2f) -> vec2f {
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

fn op_recip(lhs: vec2f) -> vec2f {
    if (lhs[0] > 0.0 || lhs[1] < 0.0) {
        return vec2f(1.0 / lhs[1], 1.0 / lhs[0]);
    } else {
        return nan_i();
    }
}

fn op_sqrt(lhs: vec2f) -> vec2f {
    if (lhs[0] >= 0.0) {
        return sqrt(lhs);
    } else {
        return nan_i();
    }
}

fn op_floor(lhs: vec2f) -> vec2f {
    return floor(lhs);
}

fn op_ceil(lhs: vec2f) -> vec2f {
    return ceil(lhs);
}

fn op_round(lhs: vec2f) -> vec2f {
    return round(lhs);
}

fn op_square(lhs: vec2f) -> vec2f {
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

fn op_sin(lhs: vec2f) -> vec2f {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return vec2f(-1.0, 1.0);
    }
}

fn op_cos(lhs: vec2f) -> vec2f {
    if (has_nan(lhs)) {
        return nan_i();
    } else {
        return vec2f(-1.0, 1.0);
    }
}

fn op_tan(lhs: vec2f) -> vec2f {
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

fn op_asin(lhs: vec2f) -> vec2f {
    if (lhs[0] < -1.0 || lhs[1] > 1.0) {
        return nan_i();
    } else {
        return asin(lhs);
    }
}

fn op_acos(lhs: vec2f) -> vec2f {
    if (lhs[0] < -1.0 || lhs[1] > 1.0) {
        return nan_i();
    } else {
        return acos(lhs).yx;
    }
}

fn op_atan(lhs: vec2f) -> vec2f {
    return atan(lhs);
}

fn op_exp(lhs: vec2f) -> vec2f {
    return exp(lhs);
}

fn op_log(lhs: vec2f) -> vec2f {
    if (lhs[0] < 0.0) {
        return nan_i();
    } else {
        return log(lhs);
    }
}

fn op_not(lhs: vec2f) -> vec2f {
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

fn op_compare(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn op_and(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn op_or(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn build_imm(v: f32) -> vec2f {
    return vec2f(v);
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if (r < 0.0) {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn op_mod(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn op_add(lhs: vec2f, rhs: vec2f) -> vec2f {
    return lhs + rhs;
}

fn op_sub(lhs: vec2f, rhs: vec2f) -> vec2f {
    return lhs - rhs.yx;
}

fn op_min(lhs: vec2f, rhs: vec2f) -> vec2f {
    return min(lhs, rhs);
}

fn op_max(lhs: vec2f, rhs: vec2f) -> vec2f {
    return max(lhs, rhs);
}

fn op_mul(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn op_div(lhs: vec2f, rhs: vec2f) -> vec2f {
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

fn op_atan2(lhs: vec2f, rhs: vec2f) -> vec2f {
    if (has_nan(lhs) || has_nan(rhs)) {
        return nan_i();
    }
    return vec2f(-3.141592654, 3.141592654);
}
