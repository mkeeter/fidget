// Initialize the tiles and choices buffer to begin rendering an image
//
// This should be invoked with a total of cfg.tiles_count threads, which
// is given by (cfg.image_size / tiles_in.tile_size) ** 2 in this case
// (not necessarily true in later stages, where we could render fewer tiles)
//
// The exact threads per group doesn't matter
//
// tiles_in and choices_in represent the inputs to the first stage of
// interval kernel evaluation; tiles_out is the output of the first stage
// of interval kernel evaluation.
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  device uint32_t* choices_in [[buffer(1)]],
                  device RenderOut& tiles_in [[buffer(2)]],
                  device RenderOut& tiles_out [[buffer(3)]],
                  uint index [[thread_position_in_grid]],
                  uint threads [[threads_per_grid]])
{
    const uint tiles_per_side = cfg.image_size / INITIAL_TILE_SIZE;
    const uint tile_count = tiles_per_side * tiles_per_side;

    if (index >= tile_count) {
        return;
    }

    const uint tile_x = index % tiles_per_side;
    const uint tile_y = index / tiles_per_side;

    if (tile_x > 0xFFFF || tile_y > 0xFFFF) {
        // If there are too many tiles, assign an obviously wrong value
        // (hard to check, alas)
        tiles_in.tiles[index] = 0xFFFFFFFF;
    } else {
        // Otherwise, assign the bit-packed tile
        tiles_in.tiles[index] = tile_x | (tile_y << 16);
    }

    // Groups of 64 tiles refer to a single choice array during interval
    // evaluation.  This is a little weird during the first evaluation, but
    // makes subdivision sensible later on.
    // that every choice should examine both branches.
    if (index % 64 == 0) {
        for (uint i=0; i < cfg.choice_buf_size; i += 1) {
            choices_in[index * cfg.choice_buf_size + i] = 0xFFFFFFFF;
        }
    }

    if (index == 0) {
        // Assign the tile count for the first render pass
        tiles_in.tile_count = tile_count;
        tiles_in.tile_size = INITIAL_TILE_SIZE;

        // Reset the tile accumulator, in case this we're reusing a buffer
        tiles_out.tile_count = 0;
        tiles_out.tile_size = INITIAL_TILE_SIZE;
    }
}
