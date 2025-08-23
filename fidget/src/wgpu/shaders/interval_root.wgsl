// Interval root tile evaluation
//
// This must be combined with opcode definitions and the generic interpreter
// from `tape_interpreter.wgsl`

@group(1) @binding(0) var<storage, read_write> tile_zmin: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_tape_tree: ListOutput;

/// Root tile size
const TILE_SIZE: u32 = 64;

@compute @workgroup_size(4, 4, 4)
fn interval_root_main(
    @builtin(global_invocation_id) tile_corner: vec3u
) {
    // Calculate render size in tile units
    let size64 = config.render_size / 64;

    // Reset this tile's position in the tape tree
    let tile_corner = global_id;
    let tile_index = tile_corner.x + 
        tile_corner.y * size64.x +
        tile_corner.z * size64.x * size64.y;
    tile_tape_tree.data[tile_index] = 0u;

    if (tile_corner.x > size64.x ||
        tile_corner.y > size64.y ||
        tile_corner.z > size64.z)
    {
        return;
    }

    // Tile corner position, in voxels
    let corner_pos = tile_corner * TILE_SIZE;

    // Compute transformed interval regions
    let m = interval_inputs(tile_corner, TILE_SIZE);

    // Check for Z masking from tile, in case one of the other tiles finished
    // first and was unambiguously filled.  This isn't likely, but the check is
    // cheap, so why not?
    let tile_index_xy = tile_corner.x + tile_corner.y * size64.x;
    if tile_zmin[tile_index_xy] >= corner_pos.z {
        return;
    }

    // Do the actual interpreter work
    var stack = Stack();
    let out = run_tape(tape_index, m, &stack)[0];

    // If the tile is completely empty, then we're done; we've already written
    // the tile tape to 0u.
    if out[0] > 0.0 {
        return;
    }

    if stack.has_choice {
        // TODO simplify tape
    }

    if out[1] < 0.0 {
        // This tile is full
        atomicMax(&tile_zmin[tile_index_xy], corner_pos.z + TILE_SIZE);
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
