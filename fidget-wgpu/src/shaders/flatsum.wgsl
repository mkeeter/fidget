// Number of tiles for each Z value (used for a quasi-radix sort)
@group(1) @binding(0) var<storage, read> tiles_hist: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_z_to_offset: array<u32>;
@group(1) @binding(2) var<storage, read_write> dispatch: Dispatch;

const TILE_SIZE: u32 = 4;
override MAX_TILES_PER_DISPATCH: u32;

/// Dispatched with one workgroup (PEAK EFFICIENCY)
@compute @workgroup_size(1, 1, 1)
fn flatsum_main() {
    let size_tiles = config.render_size / TILE_SIZE;
    var buffer_offset = 0u;
    var d = 0u;
    for (var i = 0u; i < size_tiles.z; i += 1) {
        let z = size_tiles.z - i - 1;
        tile_z_to_offset[z] = buffer_offset;
        buffer_offset += tiles_hist[z];
    }
    if buffer_offset > 0u {
        // Schedule the voxel dispatch.  It's okay to dispatch fewer workgroups
        // than we have tiles, because the voxel shader loops.
        let n = min(buffer_offset, MAX_TILES_PER_DISPATCH);
        dispatch = Dispatch(vec3u(n, 1u, 1u), buffer_offset);
    } else {
        dispatch = Dispatch(vec3u(0, 0, 0), 0);
    }
}
