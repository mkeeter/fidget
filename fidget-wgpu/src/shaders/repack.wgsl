@group(1) @binding(0) var<storage, read> tiles: TileListInput;
@group(1) @binding(1) var<storage, read_write> tiles_hist: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> tile_z_offset: array<atomic<u32>>;

/// Sorted output list of tiles
@group(1) @binding(3) var<storage, read_write> tiles_out: TileListOutput;
@group(1) @binding(4) var<storage, read_write> dispatch: array<Dispatch>;

override TILE_SIZE: u32;

const CUMSUM_FLAG: u32 = 1 << 31;
const CUMSUM_MASK: u32 = CUMSUM_FLAG - 1;

/// Gets the cumulative offset for a particular Z value
///
/// This uses a decoupled loop-back algorithm
fn cumsum(z: u32) -> u32 {
    let prev = atomicLoad(&tiles_hist[z]);
    if (prev & CUMSUM_FLAG) != 0u {
        return prev & CUMSUM_MASK;
    }
    var cumsum = 0u;
    var i = z;
    loop {
        if i == 0 {
            break;
        }
        i -= 1;
        let p = atomicLoad(&tiles_hist[i]);
        if (p & CUMSUM_FLAG) != 0u {
            cumsum += p & CUMSUM_MASK;
            break;
        } else {
            cumsum += p;
        }
    }
    let out = prev + cumsum;
    atomicStore(&tiles_hist[z], out | CUMSUM_FLAG);
    return out;
}

/// Dispatched with one thread per position in `tiles` (rounded up)
@compute @workgroup_size(64, 1, 1)
fn repack_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Calculate render size in tile units
    let size_tiles = config.render_size / TILE_SIZE;

    if global_id.x == 0 {
        tiles_out.count = tiles.count;
    } else if global_id.x >= tiles.count {
        return;
    }

    let t = tiles.active_tiles[global_id.x];
    let z = t.tile / (size_tiles.x * size_tiles.y);
    let buffer_offset = cumsum(z);
    let count = atomicAdd(&tile_z_offset[z], 1u);
    tiles_out.tiles[tiles_out.count - (buffer_offset - count)] = t;

    // One thread gets to populate the dispatch array
    if global_id.x == 0u {
        var count = cumsum(size_tiles.z - 1);
        var d = 0u;
        while count > 0 && d < arrayLength(&dispatch) {
            let n = min(config.max_tiles_per_dispatch, count);
            dispatch[d] = Dispatch(vec3u(n, 1u, 1u));
            count -= n;
            d += 1;
        }
        // Clear all subsequent dispatch stages
        while d < arrayLength(&dispatch) {
            dispatch[d] = Dispatch(vec3u(0, 0, 0));
            d += 1;
        }
    }
}
