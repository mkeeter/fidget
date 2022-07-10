// At this point, we have some number of active tiles, which is recorded in
// cfg.tile_count.  This may be the result of subdividing tiles at a previous
// step of the evaluation, but that doesn't matter for our purposes.
//
// Each tile is associated with a chunk of the choices array, which is
// read-write: previous evaluation may have written to it, and this stage
// may modify it further.
kernel void main0(const device RenderConfig& cfg [[buffer(0)]],
                  device uint32_t* tiles [[buffer(1)]],
                  device uint8_t* choices [[buffer(2)]],
                  device uint8_t* image [[buffer(3)]],
                  device RenderOut& out [[buffer(4)]],
                  uint index [[thread_position_in_grid]])
{
    // Early exit in case the kernel has a few extra threads
    if (index >= cfg.tile_count) {
        return;
    }

    // Find the upper and lower bounds of the tile, in pixel coordinates
    const uint32_t tile = tiles[index];
    const uint2 lower = cfg.tile_size * uint2(tile & 0xFFFF, tile >> 16);
    const uint2 upper = lower + cfg.tile_size;

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

    // Perform interval evaluation
    const float2 result = t_eval(vars, &choices[index * CHOICE_COUNT]);

    // Accumulate the number of active tiles
    const bool active = (result[0] <= 0.0 && result[1] >= 0.0);
    if (active) {
        // TODO: maybe separate thread-group then global accumulation?
        const uint t = atomic_fetch_add_explicit(
            &out.active_tile_count, 1, metal::memory_order_relaxed);
        // Assign the next level of the tree
        out.next[t] = tile;
        //tiles[index] = t;
    } else {
        //tiles[index] = 0xFFFFFFFF;
    }

    // If this interval is filled, color in this pixel.  `out` is a
    // mipmap-style image, where each pixel represent a tile at our current
    // scale.
    if (result[1] < 0.0) {
        const uint2 p = lower / cfg.tile_size;
        image[p.x + p.y * cfg.image_size / cfg.tile_size] = 1;
    }
}
