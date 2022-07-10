// Subdivides the result of interval evaluation
//
// This should be invoked with `cfg` representing the next render stage.
// evaluation stage.  It should be invoked with
//      prev.active_tile_count * SPLIT_RATIO**2
// threads in total, with threadgroup size of (SPLIT_RATIO**2, 1, 1)
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant RenderOutConst& prev [[buffer(1)]],
                  const constant uint32_t* choices_in [[buffer(2)]],
                  device uint32_t* subtiles [[buffer(3)]],
                  device uint32_t* choices_out [[buffer(4)]],
                  device RenderOut& out [[buffer(5)]],
                  uint index [[thread_position_in_grid]])
{
    const uint active_tile_count = prev.active_tile_count;
    const uint subtiles_per_tile = SPLIT_RATIO * SPLIT_RATIO;
    if (index >= active_tile_count * subtiles_per_tile) {
        return;
    }
    const uint tile_index = index / subtiles_per_tile;
    const uint subtile_index = index % subtiles_per_tile;

    const TileIndex tile = prev.tiles[tile_index];

    const uint tile_x = tile.tile & 0xFFFF;
    const uint tile_y = tile.tile >> 16;
    const uint subtile_x = (tile_x * SPLIT_RATIO) +
                           (subtile_index / SPLIT_RATIO);
    const uint subtile_y = (tile_y * SPLIT_RATIO) +
                           (subtile_index % SPLIT_RATIO);
    if (subtile_x > 0xFFFF || subtile_y > 0xFFFF) {
        // If there are too many subtiles, assign an obviously wrong value
        // (hard to check, alas)
        subtiles[index] = 0xFFFFFF;
    } else {
        // Otherwise, assign the bit-packed subtile
        subtiles[index] = subtile_x | (subtile_y << 16);
    }

    // Copy the choices array from the tile into the subtile
    for (uint i=0; i < cfg.choice_buf_size; ++i) {
        choices_out[index * cfg.choice_buf_size + i] =
            choices_in[tile.prev_index * cfg.choice_buf_size + i];
    }

    // Reset the tile accumulator, in case this we're reusing a buffer
    if (index == 0) {
        atomic_store_explicit(
            &out.active_tile_count, 0, metal::memory_order_relaxed);
    }
}
