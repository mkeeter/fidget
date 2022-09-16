uchar4 pixel(uint x, uint y, uint image_size,
             const constant uint8_t* image_64x64,
             const constant uint8_t* image_8x8,
             const constant uint8_t* image_1x1)
{
    switch (image_64x64[(x / 64) + (y / 64) * (image_size / 64)]) {
        case FULL:  return uchar4(255, 200, 200, 255);
        case EMPTY: return uchar4(50, 0, 0, 255);
        default: break;
    }
    switch (image_8x8[(x / 8) + (y / 8) * (image_size / 8)]) {
        case FULL:  return uchar4(200, 255, 200, 255);
        case EMPTY: return uchar4(0, 50, 0, 255);
        default: break;
    }
    switch (image_1x1[x + y * image_size]) {
        case FULL:  return uchar4(200, 200, 255, 255);
        case EMPTY: return uchar4(0, 0, 50, 255);
        default: break;
    }
    return uchar4(255, 0, 0, 255); // Invalid
}

// Merge a set of images at different resolution into a final RGBA image
kernel void main0(const constant RenderConfig& cfg [[buffer(0)]],
                  const constant uint8_t* image_64x64 [[buffer(1)]],
                  const constant uint8_t* image_8x8 [[buffer(2)]],
                  const constant uint8_t* image_1x1 [[buffer(3)]],
                  device uchar4* out [[buffer(4)]],
                  uint index [[thread_position_in_grid]])
{
    const uint pixel_x = index % cfg.image_size;
    const uint pixel_y = index / cfg.image_size;
    if (pixel_x >= cfg.image_size || pixel_y >= cfg.image_size) {
        return;
    }
    // The merged image is flipped vertically from earlier stages
    out[pixel_x + (cfg.image_size - pixel_y - 1) * cfg.image_size] =
        pixel(pixel_x, pixel_y, cfg.image_size, image_64x64, image_8x8, image_1x1);
}
