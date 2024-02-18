use crate::jit::{arch::float_slice::SIMD_WIDTH, AssemblerData, SimdSize};

pub struct FloatSliceAssembler(pub(crate) AssemblerData<[f32; SIMD_WIDTH]>);

impl SimdSize for f32 {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}
