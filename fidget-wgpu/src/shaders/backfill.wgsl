/// Backfill tile_zmin from subtile_zmin
@group(1) @binding(0) var<storage, read> subtile_zmin: array<Voxel>;
@group(1) @binding(1) var<storage, read_write> tile_zmin: array<Voxel>;

// Things to clear
@group(1) @binding(2) var<storage, read_write> tiles_out: TileListOutput;
@group(1) @binding(3) var<storage, read_write> tile_hist: array<u32>;
@group(1) @binding(4) var<storage, read_write> tile_z_offset: array<u32>;

override TILE_SIZE: u32;

// Dispatch size is one kernel per XY tile, each of which samples a 4x4 region
@compute @workgroup_size(64)
fn backfill_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    // Reset the tile count
    if global_id.x == 0u {
        tiles_out.count = 0u;
    }
    // Reset the tile Z histogram (cooperatively)
    let stride = num_workgroups.x * 64u;
    for (var i = global_id.x; i < arrayLength(&tile_hist); i += stride) {
        tile_hist[i] = 0u;
    }
    // Reset the tile Z offset (cooperatively)
    for (var i = global_id.x; i < arrayLength(&tile_z_offset); i += stride) {
        tile_z_offset[i] = 0u;
    }

    // Prepare to do the backfilling
    let SUBTILE_SIZE = TILE_SIZE / 4u;

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    let tile_count = size_tiles.x * size_tiles.y;
    if global_id.x >= tile_count {
        return;
    }

    let tile_id = global_id.x;
    let tx = tile_id % size_tiles.x;
    let ty = (tile_id / size_tiles.x) % size_tiles.y;
    let tile_corner = vec2u(tx, ty);

    var new_zmin = 0xFFFFFFFFu;
    for (var i=0u; i < 4u; i++) {
        for (var j=0u; j < 4u; j++) {
            let subtile_corner = tile_corner * 4u + vec2u(i, j);
            let subtile_index = subtile_corner.x + subtile_corner.y * size_subtiles.x;
            let v = subtile_zmin[subtile_index].value >> 20;

            // bail out immediately if a subtile isn't populated
            if v == 0 {
                return;
            }
            new_zmin = min(new_zmin, v);
        }
    }
    // It's okay to set the tape to 0 because the normal pass (which is the only
    // thing using the tape) will prioritize the higher-resolution tiles first;
    // if this tile had a higher Z value, then we wouldn't have evaluated the
    // subtile at all, so we can't have a case where the tile has a higher Z and
    // a 0-index tape.
    tile_zmin[tile_id].value = new_zmin << 20;
}
