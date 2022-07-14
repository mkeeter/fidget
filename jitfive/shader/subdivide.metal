// Subdivides the result of interval evaluation
//
// This should be invoked with `cfg` representing the next render stage.
// evaluation stage.  It should be invoked with
//      prev.active_tile_count * SPLIT_RATIO**2
// threads in total, with threadgroup size of (SPLIT_RATIO**2, 1, 1)
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant RenderOut& tiles_prev [[buffer(1)]],
                  device RenderOut& tiles_in [[buffer(2)]],
                  device RenderOut& tiles_out [[buffer(3)]],
                  uint index [[thread_position_in_grid]])
{
    const uint active_tile_count = tiles_prev.tile_count;
    const uint subtiles_per_tile = SPLIT_RATIO * SPLIT_RATIO;
    if (index >= active_tile_count * subtiles_per_tile) {
        return;
    }
    const uint tile_index = index / subtiles_per_tile;
    const uint subtile_index = index % subtiles_per_tile;

    const uint32_t tile = tiles_prev.tiles[tile_index];

    const uint tile_x = tile & 0xFFFF;
    const uint tile_y = tile >> 16;
    const uint subtile_x = (tile_x * SPLIT_RATIO) +
                           (subtile_index / SPLIT_RATIO);
    const uint subtile_y = (tile_y * SPLIT_RATIO) +
                           (subtile_index % SPLIT_RATIO);
    if (subtile_x > 0xFFFF || subtile_y > 0xFFFF) {
        // If there are too many subtiles, assign an obviously wrong value
        // (hard to check, alas)
        tiles_in.tiles[index] = 0xFFFFFF;
    } else {
        // Otherwise, assign the bit-packed subtile
        tiles_in.tiles[index] = subtile_x | (subtile_y << 16);
    }

    if (index == 0) {
        // Assign the tile count for the next render stage
        tiles_in.tile_count = active_tile_count * subtiles_per_tile;
        tiles_in.tile_size = tiles_prev.tile_size / SPLIT_RATIO;

        // Reset the accumulator for output tiles
        tiles_out.tile_count = 0;
        tiles_out.tile_size = tiles_prev.tile_size / SPLIT_RATIO;
    }
}
