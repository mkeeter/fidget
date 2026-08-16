/// Duplicate of the `GeometryPixel` type in Rust
struct GeometryPixel {
    normal: vec3f,
    depth: u32,
}

fn nan_f32() -> f32 {
  // Workaround for https://github.com/gpuweb/gpuweb/issues/3749
  let bits = 0xffffffffu;
  return bitcast<f32>(bits);
}

/// Single word in an expression tape
struct TapeWord {
    op: u32,
    imm: u32,
}

/// Dynamic list of tiles, using an atomic bump allocator
struct TileListOutput {
    wg_size: array<atomic<u32>, 3>,
    count: atomic<u32>,

    /// Flexible array member
    active_tiles: array<u32>,
}

/// Read-only version of `TileListOutput`
struct TileListInput {
    wg_size: array<u32, 3>,
    count: u32,
    active_tiles: array<u32>,
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if r < 0.0 {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn div_euclid(lhs: f32, rhs: f32) -> f32 {
    let q = trunc(lhs / rhs);
    if lhs % rhs < 0.0 {
        if rhs > 0.0 {
            return q - 1.0;
        } else {
            return q + 1.0;
        }
    } else {
        return q;
    }
}


fn hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

fn rand(seed: u32) -> f32 {
    let h = hash(seed);
    let bits = (h >> 9) | 0x3f800000;
    return bitcast<f32>(bits) - 1.0;
}

fn mix(a: u32, b: u32) -> u32 {
    return hash(a + hash(b));
}
