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
