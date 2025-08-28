/// Backfill tile_zmin from subtile_zmin
@group(1) @binding(0) var<storage, read> subtile_zmin: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_zmin: array<u32>;

// Clear tile counters
@group(1) @binding(2) var<storage, read_write> count_clear: array<u32, 4>;

override TILE_SIZE: u32;

// Dispatch size is one kernel per XY tile, each of which samples a 4x4 region
@compute @workgroup_size(64)
fn backfill_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Reset an unused counter
    if global_id.x < 4 {
        count_clear[global_id.x] = 0u;
    }

    let SUBTILE_SIZE = TILE_SIZE / 4u;

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Reset various counters to prepare for the next strata
    if TILE_SIZE == 64 && global_id.x == 0u && global_id.y == 0u && global_id.z == 0u {
        config.tape_data_offset = config.root_tape_len;
        config.tile_tapes_offset = size64.x * size64.y * size64.z;
    }

    let tile_count = size_tiles.x * size_tiles.y;
    if global_id.x >= tile_count {
        return;
    }

    let tile_id = global_id.x;
    let tx = tile_id % size_tiles.x;
    let ty = (tile_id / size_tiles.x) % size_tiles.y;
    let tile_corner = vec2u(tx, ty);

    var all_present = true;
    var new_zmin = 0xFFFFFFFFu;
    for (var i=0u; i < 4u; i++) {
        for (var j=0u; j < 4u; j++) {
            let subtile_corner = tile_corner * 4u + vec2u(i, j);
            let subtile_index = subtile_corner.x + subtile_corner.y * size_subtiles.x;
            let v = subtile_zmin[subtile_index];
            if v != 0 {
                new_zmin = min(new_zmin, v);
            } else {
                all_present = false;
            }
        }
    }
    if all_present {
        tile_zmin[tile_id] = new_zmin;
    }
}
