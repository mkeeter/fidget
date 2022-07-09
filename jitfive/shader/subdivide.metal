// Subdivides the result of interval evaluation
//
// This should be invoked with `cfg` representing the most recently completed
// evaluation stage.  It should be invoked with
//      prev.active_tiles * cfg.tile_scale**2
// threads in total, with threadgroup size of cfg.tile_scale**2
kernel void main0(const device RenderConfig& cfg [[buffer(0)]],
                  const device RenderOut& prev [[buffer(1)]],
                  const device uint8_t* choices_in [[buffer(2)]],
                  device uint32_t* subtiles [[buffer(3)]],
                  device uint8_t* choices_out [[buffer(4)]],
                  uint index [[thread_position_in_grid]])
{
    const uint subtiles_per_tile = cfg.tile_scale * cfg.tile_scale;
    if (index >= prev.active_tiles * subtiles_per_tile) {
        return;
    }
    const uint tile_index = index / subtiles_per_tile;
    const uint subtile_index = index % subtiles_per_tile;
    const uint tile = prev.next[tile_index];

    const uint tile_x = tile & 0xFFFF;
    const uint tile_y = tile >> 16;
    const uint subtile_x = tile_x * cfg.tile_scale +
                           subtile_index / cfg.tile_scale;
    const uint subtile_y = tile_y * cfg.tile_scale +
                           subtile_index % cfg.tile_scale;
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
            choices_in[tile_index * cfg.choice_count + i];
    }
}
