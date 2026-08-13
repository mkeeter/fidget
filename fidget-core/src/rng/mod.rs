//! Fast deterministic pseudo-random numbers

/// Single-input hash function
///
/// This function is sourced from "Hash Functions for GPU Rendering", Jarzynski
/// & Olano, 2020 ([PDF](https://jcgt.org/published/0009/03/02/paper.pdf))
#[inline]
pub fn hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Generates a random floating-point value from a seed
///
/// The seed is not assumed to be random; sequential integers or bitcast float
/// values will still produce well-distributed outputs.
#[inline]
pub fn rand(seed: u32) -> f32 {
    let h = hash(seed);
    let bits = (h >> 9) | 0x3f80_0000;
    f32::from_bits(bits) - 1.0
}

/// Mixes two random seeds, returning a new random value
///
/// The seeds are not assumed to be random or evenly distributed; sequential
/// integers or bitcast floats will still produce well-distributed outputs.
#[inline]
pub fn mix(a: u32, b: u32) -> u32 {
    // See section 4.2 in the above paper, discussing nested hashes
    hash(a.wrapping_add(hash(b)))
}
