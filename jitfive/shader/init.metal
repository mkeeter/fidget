// Initialize the tiles and choices buffer to begin rendering an image
//
// This should be invoked with a total of cfg.tiles_count threads, which
// is given by (cfg.image_size / cfg.tile_size) ** 2 in this case
// (not necessarily true in later stages, where we could render fewer tiles)
//
// The exact threads per group doesn't matter
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  device uint32_t* tiles [[buffer(1)]],
                  device uint8_t* choices [[buffer(2)]],
                  device RenderOut& out [[buffer(3)]],
                  uint index [[thread_position_in_grid]])
{
    if (index >= cfg.tile_count) {
        return;
    }

    const uint tiles_per_side = cfg.image_size / cfg.tile_size;
    const uint tile_x = index % tiles_per_side;
    const uint tile_y = index / tiles_per_side;

    if (tile_x > 0xFFFF || tile_y > 0xFFFF) {
        // If there are too many tiles, assign an obviously wrong value
        // (hard to check, alas)
        tiles[index] = 0xFFFFFF;
    } else {
        // Otherwise, assign the bit-packed tile
        tiles[index] = tile_x | (tile_y << 16);
    }

    // Fill this chunk of the choices array to always take both branches
    for (uint i=0; i < cfg.choice_count; ++i) {
        choices[index * cfg.choice_count + i] = LHS | RHS;
    }

    // Reset the tile accumulator, in case this we're reusing a buffer
    if (index == 0) {
        atomic_store_explicit(
            &out.active_tile_count, 0, metal::memory_order_relaxed);
    }
}
