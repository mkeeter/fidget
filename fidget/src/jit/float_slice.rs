use crate::jit::{
    arch::float_slice::SIMD_WIDTH, AssemblerData, JitEval, SimdType,
};

pub struct FloatSliceAssembler(pub(crate) AssemblerData<[f32; SIMD_WIDTH]>);

impl SimdType for *const f32 {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}

pub type JitFloatSliceEval = JitEval<*const f32>;
