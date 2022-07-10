// Subdivides the result of interval evaluation
//
// This should be invoked with `cfg` representing the most recently completed
// evaluation stage.  It should be invoked with
//      prev.active_tile_count * cfg.split_ratio**2
// threads in total, with threadgroup size of (cfg.split_ratio**2, 1, 1)
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant RenderOutConst& prev [[buffer(1)]],
                  const constant uint8_t* choices_in [[buffer(2)]],
                  device uint32_t* subtiles [[buffer(3)]],
                  device uint8_t* choices_out [[buffer(4)]],
                  uint index [[thread_position_in_grid]])
{
    const uint active_tile_count = prev.active_tile_count;
    const uint subtiles_per_tile = cfg.split_ratio * cfg.split_ratio;
    if (index >= active_tile_count * subtiles_per_tile) {
        return;
    }
    const uint tile_index = index / subtiles_per_tile;
    const uint subtile_index = index % subtiles_per_tile;

    const TileIndex tile = prev.tiles[tile_index];

    const uint tile_x = tile.tile & 0xFFFF;
    const uint tile_y = tile.tile >> 16;
    const uint subtile_x = (tile_x * cfg.split_ratio) +
                           (subtile_index / cfg.split_ratio);
    const uint subtile_y = (tile_y * cfg.split_ratio) +
                           (subtile_index % cfg.split_ratio);
    if (subtile_x > 0xFFFF || subtile_y > 0xFFFF) {
        // If there are too many subtiles, assign an obviously wrong value
        // (hard to check, alas)
        subtiles[index] = 0xFFFFFF;
    } else {
        // Otherwise, assign the bit-packed subtile
        subtiles[index] = subtile_x | (subtile_y << 16);
    }

    // Copy the choices array from the tile into the subtile
    for (uint i=0; i < cfg.choice_count; ++i) {
        choices_out[index * cfg.choice_count + i] =
            choices_in[tile.prev_index * cfg.choice_count + i];
    }
}
