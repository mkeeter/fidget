// At this point, we have some number of active tiles, which is recorded in
// tiles_in.tile_count.  Tiles in tiles_in.tiles are grouped into
// groups of size SPLIT_RATIO**2; that group index is used to index into
// choices_in.
//
// Active tiles are written to tiles_out.tiles (without subdivision), and their
// choices are written to CHOICE_BUF_SIZE chunks of choices_out.
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant RenderOut& tiles_in [[buffer(1)]],
                  const constant uint32_t* choices_in [[buffer(2)]],
                  device RenderOutAtomic& tiles_out [[buffer(3)]],
                  device uint32_t* choices_out [[buffer(4)]],
                  device uint8_t* image [[buffer(5)]],
                  uint index [[thread_position_in_grid]])
{
    // Early exit in case the kernel has a few extra threads
    if (index >= tiles_in.tile_count) {
        return;
    }

    // Find the upper and lower bounds of the tile, in pixel coordinates
    const uint32_t tile = tiles_in.tiles[index];
    const uint2 lower = tiles_in.tile_size * uint2(tile & 0xFFFF, tile >> 16);
    const uint2 upper = lower + tiles_in.tile_size;

    // Image location (-1 to 1)
    const float2 lower_f = cfg.pixel_to_pos(lower);
    const float2 upper_f = cfg.pixel_to_pos(upper);

    // Inject X and Y into local (thread) variables array
    float2 vars[VAR_COUNT];
    if (cfg.var_index_x < VAR_COUNT) {
        vars[cfg.var_index_x] = float2(lower_f.x, upper_f.x);
    }
    if (cfg.var_index_y < VAR_COUNT) {
        vars[cfg.var_index_y] = float2(lower_f.y, upper_f.y);
    }

    // We'll accumulate into a local choices array, then copy it to the output
    // array if we're splitting this tile.
    uint32_t local_choices[CHOICE_BUF_SIZE];

    // Perform interval evaluation!
    // Each groups of 8x8 shares a common choice tape from the previous stage
    const float2 result = t_eval(
        vars,
        &choices_in[index / (SPLIT_RATIO * SPLIT_RATIO) * CHOICE_BUF_SIZE],
        local_choices);

    // Accumulate the number of active tiles
    const bool active = (result[0] <= 0.0 && result[1] >= 0.0);
    if (active) {
        // TODO: maybe separate thread-group then global accumulation?
        // simd_prefix_exclusive_sum is useful here
        const uint t = atomic_fetch_add_explicit(
            &tiles_out.tile_count, 1, metal::memory_order_relaxed);

        // Assign the next level of the tree (without subdivision)
        tiles_out.tiles[t] = tile;

        for (uint i=0; i < CHOICE_BUF_SIZE; ++i) {
            choices_out[t * CHOICE_BUF_SIZE + i] = local_choices[i];
        }
    } else {
        // If this interval is filled or empty, color in this pixel.  `out` is
        // a mipmap-style image, where each pixel represent a tile at our
        // current scale.
        const uint2 p = lower / tiles_in.tile_size;
        if (result[1] < 0.0) {
            image[p.x + p.y * cfg.image_size / tiles_in.tile_size] = FULL;
        } else if (result[0] > 0.0) {
            image[p.x + p.y * cfg.image_size / tiles_in.tile_size] = EMPTY;
        }
    }
}
