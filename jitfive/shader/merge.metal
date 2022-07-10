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
    const uint i = pixel_x + (cfg.image_size - pixel_y - 1) * cfg.image_size;
    switch (image_64x64[(pixel_x / 64) + (pixel_y / 64) * (cfg.image_size / 64)]) {
        case FULL:  out[i] = uchar4(255, 200, 200, 255); return;
        case EMPTY: out[i] = uchar4(50, 0, 0, 255); return;
        default: break;
    }
    switch (image_8x8[(pixel_x / 8) + (pixel_y / 8) * (cfg.image_size / 8)]) {
        case FULL:  out[i] = uchar4(200, 255, 200, 255); return;
        case EMPTY: out[i] = uchar4(0, 50, 0, 255); return;
        default: break;
    }
    switch (image_1x1[index]) {
        case FULL:  out[i] = uchar4(200, 200, 255, 255); return;
        case EMPTY: out[i] = uchar4(0, 0, 50, 255); return;
        default: break;
    }
    out[i] = uchar4(255, 0, 0, 255); // Invalid
}
