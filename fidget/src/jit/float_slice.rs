use crate::jit::{arch::float_slice::SIMD_WIDTH, JitEval, SimdType};

pub struct FloatSliceAssembler;

impl SimdType for *const f32 {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}

pub type JitFloatSliceEval = JitEval<*const f32>;
