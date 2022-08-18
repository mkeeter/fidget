#include <cstdint>
#include <cmath>
#include <atomic>

using std::atomic_uint32_t;
struct uint2 {
    uint32_t x;
    uint32_t y;
    const uint32_t& operator[](int i) const {
        if (i == 0) {
            return x;
        } else {
            return y;
        }
    }
    uint2(uint32_t a, uint32_t b) {
        x = a;
        y = b;
    }
    uint2 operator+(const uint2& rhs) const {
        return uint2((*this)[0] + rhs[0], (*this)[1] + rhs[1]);
    }
    uint2 operator+(const uint32_t& rhs) const {
        return uint2((*this)[0] + rhs, (*this)[1] + rhs);
    }
    uint2 operator/(const uint32_t& rhs) const {
        return uint2((*this)[0] / rhs, (*this)[1] / rhs);
    }
};
uint2 operator*(uint32_t lhs, const uint2& rhs) {
    return uint2(lhs * rhs[0], lhs * rhs[1]);
}
struct float2 {
    float x;
    float y;
    float2() {
        // YOLO
    }
    float2(float a, float b) {
        x = a;
        y = b;
    }
    float2(float a) {
        x = a;
        y = a;
    }
    float2(uint2 a) {
        x = a[0];
        y = a[1];
    }
    const float& operator[](int i) const {
        if (i == 0) {
            return x;
        } else {
            return y;
        }
    }
    float2 operator/(const float2& rhs) const {
        return float2((*this)[0] / rhs[0], (*this)[1] / rhs[1]);
    }
    float2 operator*(const float& rhs) const {
        return float2((*this)[0] * rhs, (*this)[1] * rhs);
    }
    float2 operator+(const float2& rhs) const {
        return float2((*this)[0] + rhs[0], (*this)[1] + rhs[1]);
    }
    float2 operator-(const float& rhs) const {
        return float2((*this)[0] - rhs, (*this)[1] - rhs);
    }
};
namespace metal {
    float fmin(float a, float b) {
        return fminf(a, b);
    }
    float2 fmin(float2 a, float2 b) {
        return float2(fminf(a[0], b[0]), fminf(a[1], b[1]));
    }
    float fmax(float a, float b) {
        return fmaxf(a, b);
    }
    float2 fmax(float2 a, float2 b) {
        return float2(fmaxf(a[0], b[0]), fmaxf(a[1], b[1]));
    }
    typedef atomic_uint32_t atomic_uint;
    float sqrt(float a) {
        return sqrtf(a);
    }
    inline constexpr std::memory_order memory_order_relaxed = std::memory_order::relaxed;
}
typedef uint32_t uint;

#define constant
#define device
#define thread
#define kernel
