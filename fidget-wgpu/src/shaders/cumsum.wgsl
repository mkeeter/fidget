// Number of tiles for each Z value (used for a quasi-radix sort)
@group(1) @binding(0) var<storage, read> tiles_hist: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_z_to_offset: array<u32>;
@group(1) @binding(2) var<storage, read_write> dispatch: array<Dispatch>;

override TILE_SIZE: u32;
override MAX_DISPATCH: u32;

/// Dispatched with one workgroup (PEAK EFFICIENCY)
@compute @workgroup_size(1, 1, 1)
fn cumsum_main() {
    let size_tiles = config.render_size / TILE_SIZE;
    var cumsum = 0u;
    var dispatch_buffer_offset = 0u;
    var buffer_offset = 0u;
    var d = 0u;
    for (var i = 0u; i < size_tiles.z; i += 1) {
        let z = size_tiles.z - i - 1;
        tile_z_to_offset[z] = buffer_offset;
        if (cumsum + tiles_hist[z] > MAX_DISPATCH) {
            dispatch[d] = plan_dispatch(cumsum, dispatch_buffer_offset);
            d += 1;
            dispatch_buffer_offset += cumsum;
            cumsum = 0;
        } else {
            cumsum += tiles_hist[z];
        }
        buffer_offset += tiles_hist[z];
    }
    if cumsum > 0u {
        // schedule the final dispatch
        dispatch[d] = plan_dispatch(cumsum, dispatch_buffer_offset);
        d += 1;
    }

    // clear all subsequent dispatch stages
    while d < arrayLength(&dispatch) {
        dispatch[d] = Dispatch(vec3u(0, 0, 0), 0, 0);
        d += 1;
    }
}

fn plan_dispatch(count: u32, buffer_offset: u32) -> Dispatch {
    let n = min(count, MAX_DISPATCH);
    return Dispatch(vec3u(n, 1u, 1u), count, buffer_offset);
}
