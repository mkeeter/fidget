@group(1) @binding(0) var<storage, read> tiles: TileListInput;
@group(1) @binding(1) var<storage, read> tile_z_to_dispatch: array<u32>;
@group(1) @binding(2) var<storage, read> dispatch: array<Dispatch>;

/// Scratch buffer used to pack tiles
@group(1) @binding(3) var<storage, read_write> z_scratch: array<atomic<u32>>;

/// Sorted output list of tiles
@group(1) @binding(4) var<storage, read_write> tiles_out: array<ActiveTile>;

override TILE_SIZE: u32;

/// Dispatched with one thread per position in `tiles`
@compute @workgroup_size(64, 1, 1)
fn repack_main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Calculate render size in tile units
    let size_tiles = config.render_size / TILE_SIZE;

    if global_id.x >= tiles.count {
        return;
    }

    let t = tiles.active_tiles[global_id.x];
    let z = t.tile / (size_tiles.x * size_tiles.y);
    var d = tile_z_to_dispatch[z];
    let buffer_offset = dispatch[d].buffer_offset;
    let count = atomicAdd(&z_scratch[d], 1u);
    tiles_out[buffer_offset + count] = t;
}
