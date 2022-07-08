// Prelude
#include <metal_stdlib>

#define RHS 1
#define LHS 2

// This must be kept in sync with the Rust `struct RenderConfig`!
struct RenderConfig {
    uint32_t image_size;
    uint32_t tile_size;
    uint32_t tile_count;
    uint32_t var_index_x;
    uint32_t var_index_y;
    uint32_t var_index_z;

    // Converts from a pixel position to a floating-point image position
    float2 pixel_to_pos(uint2 pixel) const device {
        return float2(pixel) / float2(image_size - 1) * 2.0 - 1.0;
    }
};

// Floating-point math
inline float f_mul(const float a, const float b) {
    return a * b;
}
inline float f_add(const float a, const float b) {
    return a + b;
}

inline float f_min(const float a, const float b) {
    return metal::fmin(a, b);
}
inline float f_max(const float a, const float b) {
    return metal::fmax(a, b);
}
inline float f_neg(const float a) {
    return -a;
}
inline float f_sqrt(const float a) {
    return metal::sqrt(a);
}
inline float f_const(const float a) {
    return a;
}
inline float f_var(const float a) {
    return a;
}

// Interval math
inline float2 i_mul(const float2 a, const float2 b) {
    if (a[0] < 0.0f) {
        if (a[1] > 0.0f) {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // M * M
                    return float2(metal::fmin(a[0] * b[1], a[1] * b[0]),
                                  metal::fmax(a[0] * b[0], a[1] * b[1]));
                } else { // M * N
                    return float2(a[1] * b[0], a[0] * b[0]);
                }
            } else {
                if (b[1] > 0.0f) { // M * P
                    return float2(a[0] * b[1], a[1] * b[1]);
                } else { // M * Z
                    return float2(0.0f, 0.0f);
                }
            }
        } else {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // N * M
                    return float2(a[0] * b[1], a[0] * b[0]);
                } else { // N * N
                    return float2(a[1] * b[1], a[0] * b[0]);
                }
            } else {
                if (b[1] > 0.0f) { // N * P
                    return float2(a[0] * b[1], a[1] * b[0]);
                } else { // N * Z
                    return float2(0.0f, 0.0f);
                }
            }
        }
    } else {
        if (a[1] > 0.0f) {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // P * M
                    return float2(a[1] * b[0], a[1] * b[1]);
                } else {// P * N
                    return float2(a[1] * b[0], a[0] * b[1]);
                }
            } else {
                if (b[1] > 0.0f) { // P * P
                    return float2(a[0] * b[0], a[1] * b[1]);
                } else {// P * Z
                    return float2(0.0f, 0.0f);
                }
            }
        } else { // Z * ?
            return float2(0.0f, 0.0f);
        }
    }
}
inline float2 i_add(const float2 a, const float2 b) {
    return a + b;
}
inline float2 i_min(const float2 a, const float2 b) {
    return metal::fmin(a, b);
}
inline float2 i_max(const float2 a, const float2 b) {
    return metal::fmax(a, b);
}
inline float2 i_neg(const float2 a) {
    return float2(-a[1], -a[0]);
}
inline float2 i_sqrt(const float2 a) {
    if (a[1] < 0.0) {
        return float2(-1e8, 1e8); // XXX
    } else if (a[0] <= 0.0) {
        return float2(0.0, metal::sqrt(a[1]));
    } else {
        return float2(metal::sqrt(a[0]), metal::sqrt(a[1]));
    }
}
inline float2 i_const(const float a) {
    return float2(a, a);
}
inline float2 i_var(const float2 a) {
    return a;
}
