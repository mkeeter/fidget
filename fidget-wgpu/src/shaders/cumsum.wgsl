// Number of tiles for each Z value (used for a quasi-radix sort)
@group(1) @binding(0) var<storage, read> tiles_hist: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_z_to_offset: array<u32>;
@group(1) @binding(2) var<storage, read_write> dispatch: array<Dispatch>;

override TILE_SIZE: u32;
override MAX_TILES_PER_DISPATCH: u32;

/// Dispatched with one workgroup (PEAK EFFICIENCY)
@compute @workgroup_size(1, 1, 1)
fn cumsum_main() {
    // For each Z value, compute a buffer offset to be used when repacking
    // This is just a cumulative sum of tiles per Z value
    let size_tiles = config.render_size / TILE_SIZE;
    var buffer_offset = 0u;
    for (var i = 0u; i < size_tiles.z; i += 1) {
        let z = size_tiles.z - i - 1;
        tile_z_to_offset[z] = buffer_offset;
        buffer_offset += tiles_hist[z];
    }

    // Plan enough dispatches to process every tile
    //
    // In the voxel case (where we have only 1 output dispatch), then we'll
    // write it to MAX_TILES_PER_DISPATCH because the voxel kernel itself does
    // looping.
    var d = 0u;
    while buffer_offset > 0 && d < arrayLength(&dispatch) {
        let n = min(MAX_TILES_PER_DISPATCH, buffer_offset);
        dispatch[d] = Dispatch(vec3u(n, 1u, 1u), n);
        buffer_offset -= n;
        d += 1;
    }
    // Clear all subsequent dispatch stages
    while d < arrayLength(&dispatch) {
        dispatch[d] = Dispatch(vec3u(0, 0, 0), 0);
        d += 1;
    }
}
