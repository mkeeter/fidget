@group(1) @binding(0) var<storage, read> subtiles_out: TileListInput;
@group(1) @binding(1) var<storage, read_write> z_hist: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> sorted_subtiles: TileListOutput;

/// Output tile size
override SUBTILE_SIZE: u32;

@compute @workgroup_size(64, 1, 1)
fn sort_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let index = global_id.x +
        global_id.y * num_workgroups.x * 64u +
        global_id.z * num_workgroups.x * num_workgroups.y * 64u;
    if index >= subtiles_out.count {
        return;
    }

    // Calculate render size in tile units
    let size = config.render_size / SUBTILE_SIZE;

    // Read the tile and clear the filled bit
    let t = subtiles_out.active_tiles[index] & 0x7FFFFFFF;

    // Unpack into a Z component
    let tz = t / (size.x * size.y);

    let z_rel = tz % (64 / SUBTILE_SIZE);

    let pos = atomicSub(&z_hist[z_rel], 1u) - 1u;
    sorted_subtiles.active_tiles[pos] = t;

    let count = atomicAdd(&sorted_subtiles.count, 1u) + 1;

    // We dispatch a maximum of [32768, 1, 1] and iterate in the shader
    let wg_dispatch_x = min(count, 32768u);
    atomicMax(&sorted_subtiles.wg_size[0], wg_dispatch_x);
    atomicMax(&sorted_subtiles.wg_size[1], 1u);
    atomicMax(&sorted_subtiles.wg_size[2], 1u);
}
