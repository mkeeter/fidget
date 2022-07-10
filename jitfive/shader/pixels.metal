// This should be called with a 1D grid of size
//      (active_tiles, 1, 1)
// and with a threadgroup size of
//      (cfg.tile_size ** 2, 1, 1).
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant RenderOutConst& prev [[buffer(1)]],
                  const constant uint8_t* choices [[buffer(2)]],
                  device uint8_t* image [[buffer(3)]],
                  uint index [[thread_position_in_grid]])
{
    // We use this tile count instead of the one in the RenderConfig, which
    // is left over from the last pass of interval evaluation
    const uint tile_count = prev.active_tile_count;

    const uint32_t pixels_per_tile = cfg.tile_size * cfg.tile_size;
    if (index >= tile_count * pixels_per_tile) {
        return;
    }

    // Calculate the corner position of this tile, in pixels
    const uint32_t tile_index = index / pixels_per_tile;
    const TileIndex tile = prev.tiles[tile_index];
    const uint2 tile_corner =
        cfg.tile_size * uint2(tile.tile & 0xFFFF, tile.tile >> 16);

    // Calculate the offset within the tile, again in pixels
    const uint32_t offset = index % pixels_per_tile;
    const uint2 tile_offset(offset % cfg.tile_size, offset / cfg.tile_size);

    // Absolute pixel position
    const uint2 pixel = tile_corner + tile_offset;

    // Image location (-1 to 1)
    const float2 pos = cfg.pixel_to_pos(pixel);

    // Inject X and Y into local (thread) variables array
    float vars[VAR_COUNT];
    if (cfg.var_index_x < VAR_COUNT) {
        vars[cfg.var_index_x] = pos.x;
    }
    if (cfg.var_index_y < VAR_COUNT) {
        vars[cfg.var_index_y] = pos.y;
    }

    const float result =
        t_eval(vars, &choices[tile.prev_index * CHOICE_COUNT]);

    const uint8_t v = result < 0.0 ? FULL : EMPTY;

    image[pixel.x + pixel.y * cfg.image_size] = v;
}

