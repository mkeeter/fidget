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
