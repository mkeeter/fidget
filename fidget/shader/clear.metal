// Merge a set of images at different resolution into a final RGBA image
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  device uint8_t* image_64x64 [[buffer(1)]],
                  device uint8_t* image_8x8 [[buffer(2)]],
                  device uint8_t* image_1x1 [[buffer(3)]],
                  uint index [[thread_position_in_grid]])
{
    // Pixel coordinates in the final image
    const uint x = index % cfg.image_size;
    const uint y = index / cfg.image_size;
    if (x >= cfg.image_size || y >= cfg.image_size) {
        return;
    }
    if (x % 64 == 0 && y % 64 == 0) {
        image_64x64[(x / 64) + (y / 64) * (cfg.image_size / 64)] = 0;
    }
    if (x % 8 == 0 && y % 8 == 0) {
        image_8x8[(x / 8) + (y / 8) * (cfg.image_size / 8)] = 0;
    }
    image_1x1[x + y * cfg.image_size] = 0;
}
