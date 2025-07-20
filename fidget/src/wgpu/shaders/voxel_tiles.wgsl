// VM interpreter for floating-point values, using voxel tiles

/// Render configuration and tape data
@group(0) @binding(0) var<uniform> config: Config;

@group(0) @binding(1) var<storage, read> tiles_in: TileListInput;
@group(0) @binding(2) var<storage, read> tile4_zmin: array<u32>;

/// Output array, image size
@group(0) @binding(3) var<storage, read_write> result: array<atomic<u32>>;

/// Count to clear (unused in this pass)
@group(0) @binding(4) var<storage, read_write> count_clear: array<atomic<u32>, 4>;

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
    let size64 = config.render_size / 64;
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
    let pixel_index_xy = corner_pos.x + corner_pos.y * config.render_size.x;
    if (tile4_zmin[tile4_index_xy] >= corner_pos.z) {
        atomicMax(&result[pixel_index_xy], tile4_zmin[tile4_index_xy]);
        return;
    }

    // Last chance to bail out
    if (atomicLoad(&result[pixel_index_xy]) >= u32(corner_pos.z)) {
        return;
    }

    // Voxel evaluation: start by building up the input map
    var m = array(0.0, 0.0, 0.0);

    let pos_pixels = vec4f(vec3f(corner_pos), 1.0);
    let pos_model = config.mat * pos_pixels;

    if (config.axes.x < 3) {
        m[config.axes.x] = pos_model.x / pos_model.w;
    }
    if (config.axes.y < 3) {
        m[config.axes.y] = pos_model.y / pos_model.w;
    }
    if (config.axes.z < 3) {
        m[config.axes.z] = pos_model.z / pos_model.w;
    }

    // Do the actual interpreter work
    let out = run_tape(m)[0];

    if (out < 0.0) {
        atomicMax(&result[pixel_index_xy], corner_pos.z);
    }
}

fn build_imm(v: f32) -> f32 {
    return v;
}

fn op_abs(lhs: f32) -> f32 {
    return abs(lhs);
}

fn op_acos(lhs: f32) -> f32 {
    return acos(lhs);
}

fn op_cos(lhs: f32) -> f32 {
    return cos(lhs);
}

fn op_asin(lhs: f32) -> f32 {
    return asin(lhs);
}

fn op_atan(lhs: f32) -> f32 {
    return atan(lhs);
}

fn op_ceil(lhs: f32) -> f32 {
    return ceil(lhs);
}

fn op_floor(lhs: f32) -> f32 {
    return floor(lhs);
}

fn op_log(lhs: f32) -> f32 {
    return log(lhs);
}

fn op_recip(lhs: f32) -> f32 {
    return 1.0 / lhs;
}

fn op_round(lhs: f32) -> f32 {
    return round(lhs);
}

fn op_sin(lhs: f32) -> f32 {
    return sin(lhs);
}

fn op_tan(lhs: f32) -> f32 {
    return tan(lhs);
}

fn op_exp(lhs: f32) -> f32 {
    return exp(lhs);
}

fn op_add(lhs: f32, rhs: f32) -> f32 {
    return lhs + rhs;
}

fn op_neg(lhs: f32) -> f32 {
    return -lhs;
}

fn op_sub(lhs: f32, rhs: f32) -> f32 {
    return lhs - rhs;
}

fn op_mul(lhs: f32, rhs: f32) -> f32 {
    return lhs * rhs;
}

fn op_div(lhs: f32, rhs: f32) -> f32 {
    return lhs / rhs;
}

fn op_min(lhs: f32, rhs: f32) -> f32 {
    return min(lhs, rhs);
}

fn op_max(lhs: f32, rhs: f32) -> f32 {
    return max(lhs, rhs);
}

fn op_square(lhs: f32) -> f32 {
    return lhs * lhs;
}

fn op_sqrt(lhs: f32) -> f32 {
    return sqrt(lhs);
}

fn op_compare(lhs: f32, rhs: f32) -> f32 {
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

fn op_and(lhs: f32, rhs: f32) -> f32 {
    if lhs == 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn op_or(lhs: f32, rhs: f32) -> f32 {
    if lhs != 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn op_not(lhs: f32) -> f32 {
    return f32(lhs != 0.0);
}

fn op_mod(lhs: f32, rhs: f32) -> f32 {
    var out = lhs % rhs;
    out -= rhs * min(0.0, floor(out / rhs));
    return out;
}
