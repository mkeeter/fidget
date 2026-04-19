/// Backfill tile_zmin from subtile_zmin
@group(1) @binding(0) var<storage, read> subtile_zmin: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_zmin: array<u32>;

// Clear tile counters
@group(1) @binding(2) var<storage, read_write> count_clear: array<u32, 4>;
@group(1) @binding(3) var<storage, read_write> sort_clear: array<u32, 4>;
@group(1) @binding(4) var<storage, read_write> z_hist: array<u32>;

override TILE_SIZE: u32;

// Dispatch is X-only, one thread per XY tile
//
// Each thread samples a 4x4 region (using nested loops), no cooperation
@compute @workgroup_size(64)
fn backfill_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    // Reset counters from the previous stage
    for (var i = global_id.x; i < 4; i += num_workgroups.x * 64u) {
        count_clear[global_id.x] = 0u;
        sort_clear[global_id.x] = 0u;
    }

    // Reset the histogram from the previous stage
    for (var i = global_id.x; i < arrayLength(&z_hist); i += num_workgroups.x * 64u) {
        z_hist[i] = 0u;
    }

    let SUBTILE_SIZE = TILE_SIZE / 4u;

    // Convert to a size in tile units
    let size64 = config.render_size / 64;
    let size_tiles = size64 * (64 / TILE_SIZE);
    let size_subtiles = size_tiles * 4u;

    // Reset various counters to prepare for the next strata
    if TILE_SIZE == 64 && global_id.x == 0u {
        atomicStore(&config.tape_data_offset, atomicLoad(&config.root_tape_len));
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
