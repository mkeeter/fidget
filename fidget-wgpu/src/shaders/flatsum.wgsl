// Number of tiles for each Z value (used for a quasi-radix sort)
@group(1) @binding(0) var<storage, read> tiles_hist: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_z_to_offset: array<u32>;
@group(1) @binding(2) var<storage, read_write> dispatch: VoxelDispatch;

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
        // Schedule the voxel dispatch
        dispatch = plan_dispatch(buffer_offset);
    } else {
        dispatch = VoxelDispatch(vec3u(0, 0, 0), 0);
    }
}

fn plan_dispatch(count: u32) -> VoxelDispatch {
    let n = min(count, MAX_TILES_PER_DISPATCH);
    return VoxelDispatch(
        vec3u(min(n, 32768u), (n + 32767u) / 32768u, 1u),
        count,
    );
}
