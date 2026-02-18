@group(1) @binding(0) var<storage, read> tiles: TileListInput;
@group(1) @binding(1) var<storage, read> tile_z_to_offset: array<u32>;

/// Scratch buffer used to pack tiles
///
/// This is the same buffer as the Z histogram shader, which we subtract from to
/// get back down to 0
@group(1) @binding(2) var<storage, read_write> z_hist: array<atomic<u32>>;

/// Sorted output list of tiles
@group(1) @binding(3) var<storage, read_write> tiles_out: TileListOutput;

override TILE_SIZE: u32;

/// Dispatched with one thread per position in `tiles`
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
    var buffer_offset = tile_z_to_offset[z];
    let count = atomicSub(&z_hist[z], 1u) - 1;
    tiles_out.tiles[buffer_offset + count] = t;
}
