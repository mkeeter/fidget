// At this point, we have some number of active tiles, which is recorded in
// cfg.tile_count.  This may be the result of subdividing tiles at a previous
// step of the evaluation, but that doesn't matter for our purposes.
//
// Each tile is associated with a chunk of the choices array, which is
// read-write: previous evaluation may have written to it, and this stage
// may modify it further.
kernel void main0(const device RenderConfig& cfg [[buffer(0)]],
                  const device uint32_t* tiles [[buffer(1)]],
                  device uint8_t* choices [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    // Early exit in case the kernel has a few extra threads
    if (index >= cfg.tile_count) {
        return;
    }

    // Find the upper and lower bounds of the tile, in pixel coordinates
    const uint32_t tile = tiles[index];
    const uint2 lower = cfg.tile_size * uint2(tile & 0xFFFF, tile >> 16);
    const uint2 upper = lower + cfg.tile_scale;

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

    // Perform interval evaluation, writing to the choices array
    const float2 result = t_eval(vars, &choices[index * CHOICE_COUNT]);

    result[index] = t_eval(&vars[index * VAR_COUNT],
                           &choices[index * CHOICE_COUNT]);
}
