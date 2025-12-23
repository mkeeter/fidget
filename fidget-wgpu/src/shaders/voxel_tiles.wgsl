// VM interpreter for floating-point values, using voxel tiles

@group(1) @binding(0) var<storage, read> tiles_in: TileListInput;
@group(1) @binding(1) var<storage, read> tile4_zmin: array<u32>;

/// Output array, image size
@group(1) @binding(2) var<storage, read_write> result: array<atomic<u32>>;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Tile index is packed into two words of the workgroup ID, due to dispatch
    // size limits on any single dimension.
    let active_tile4_index = workgroup_id.x + workgroup_id.y * 32768;
    if active_tile4_index >= tiles_in.count {
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
    if tile4_zmin[tile4_index_xy] >= corner_pos.z {
        atomicMax(&result[pixel_index_xy], tile4_zmin[tile4_index_xy]);
        return;
    }

    // Last chance to bail out
    if atomicLoad(&result[pixel_index_xy]) >= u32(corner_pos.z) {
        return;
    }

    // Compute input values
    let m = transformed_inputs(
        Value(f32(corner_pos.x)),
        Value(f32(corner_pos.y)),
        Value(f32(corner_pos.z)),
    );

    // Do the actual interpreter work
    let tape_start = get_tape_start(corner_pos);
    var stack = Stack(); // dummy value
    let out = run_tape(tape_start.index, m, &stack);

    if out.value.v < 0.0 {
        atomicMax(&result[pixel_index_xy], corner_pos.z);
    }
}

struct Value {
    v: f32,
}

fn build_imm(imm: f32) -> Value {
    return Value(imm);
}

fn op_abs(lhs: Value) -> Value {
    return Value(abs(lhs.v));
}

fn op_acos(lhs: Value) -> Value {
    return Value(acos(lhs.v));
}

fn op_cos(lhs: Value) -> Value {
    return Value(cos(lhs.v));
}

fn op_asin(lhs: Value) -> Value {
    return Value(asin(lhs.v));
}

fn op_atan(lhs: Value) -> Value {
    return Value(atan(lhs.v));
}

fn op_ceil(lhs: Value) -> Value {
    return Value(ceil(lhs.v));
}

fn op_floor(lhs: Value) -> Value {
    return Value(floor(lhs.v));
}

fn op_log(lhs: Value) -> Value {
    return Value(log(lhs.v));
}

fn op_recip(lhs: Value) -> Value {
    return Value(1.0 / lhs.v);
}

fn op_round(lhs: Value) -> Value {
    return Value(round(lhs.v));
}

fn op_sin(lhs: Value) -> Value {
    return Value(sin(lhs.v));
}

fn op_tan(lhs: Value) -> Value {
    return Value(tan(lhs.v));
}

fn op_exp(lhs: Value) -> Value {
    return Value(exp(lhs.v));
}

fn op_add(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v + rhs.v);
}

fn op_neg(lhs: Value) -> Value {
    return Value(-lhs.v);
}

fn op_sub(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v - rhs.v);
}

fn op_mul(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v * rhs.v);
}

fn op_div(lhs: Value, rhs: Value) -> Value {
    return Value(lhs.v / rhs.v);
}

fn op_atan2(lhs: Value, rhs: Value) -> Value {
    return Value(atan2(lhs.v, rhs.v));
}

fn op_min(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    return Value(min(lhs.v, rhs.v));
}

fn op_max(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    return Value(max(lhs.v, rhs.v));
}

fn op_square(lhs: Value) -> Value {
    return Value(lhs.v * lhs.v);
}

fn op_sqrt(lhs: Value) -> Value {
    return Value(sqrt(lhs.v));
}

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if lhs.v < rhs.v {
        return Value(-1.0);
    } else if lhs.v > rhs.v {
        return Value(1.0);
    } else if lhs.v == rhs.v {
        return Value(0.0);
    } else {
        return Value(nan_f32());
    }
}

fn op_and(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v == 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v != 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_not(lhs: Value) -> Value {
    return Value(f32(lhs.v != 0.0));
}

fn op_mod(lhs: Value, rhs: Value) -> Value {
    var out = lhs.v % rhs.v;
    out -= rhs.v * min(0.0, floor(out / rhs.v));
    return Value(out);
}
