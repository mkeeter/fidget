// VM interpreter for floating-point values, using voxel tiles

@group(1) @binding(0) var<storage, read> tape_data: TapeData;
@group(1) @binding(1) var<storage, read> tile4_in: TileListInput;
@group(1) @binding(2) var<storage, read> tile4_zmin: array<Voxel>;

/// Output array, image size (rounded up to tile count)
@group(1) @binding(3) var<storage, read_write> result: array<Voxel>;

@compute @workgroup_size(4, 4, 4)
fn voxel_ray_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    var tile_index = workgroup_id.x; // always a 1D dispatch
    let stride = num_workgroups.x;

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size16 = size64 * 4u;
    let size4 = size16 * 4u;
    while tile_index < tile4_in.count {
        // Get global tile position, in tile4 coordinates
        let tile = tile4_in.active_tiles[tile_index];
        let t = tile.tile;
        tile_index += stride;
        let tx = t % size4.x;
        let ty = (t / size4.x) % size4.y;
        let tz = (t / (size4.x * size4.y)) % size4.z;
        let tile4_corner = vec3u(tx, ty, tz);

        // Subtile corner position, in voxels
        let corner_pos = tile4_corner * 4 + local_id;

        let tile4_index_xy = tx + ty * size4.x;
        let pixel_index_xy = corner_pos.x + corner_pos.y * config.render_size[0];
        let tile4_value = tile4_zmin[tile4_index_xy].value;
        if (tile4_value >> 20) >= corner_pos.z {
            atomicMax(&result[pixel_index_xy].value, tile4_value);
            continue;
        }

        // Last chance to bail out, if a different thread rendered this pixel
        if (atomicLoad(&result[pixel_index_xy].value) >> 20) >= corner_pos.z {
            continue;
        }

        // Compute input values
        let m = transformed_inputs(
            Value(f32(corner_pos.x)),
            Value(f32(corner_pos.y)),
            Value(f32(corner_pos.z)),
        );

        // Do the actual interpreter work
        var stack = Stack(); // dummy value
        let out = run_tape(tile.tape_index, m, &stack);

        if out.value.v < 0.0 {
            let new_z = corner_pos.z;
            let new_value = (new_z << 20) | tile.tape_index;
            atomicMax(&result[pixel_index_xy].value, new_value);
        }
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
