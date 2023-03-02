use crate::jit::{
    arch::float_slice::SIMD_WIDTH, AssemblerData, JitBulkEval, SimdAssembler,
};

pub struct FloatSliceAssembler(pub(crate) AssemblerData<[f32; SIMD_WIDTH]>);

impl SimdAssembler for FloatSliceAssembler {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}

pub type JitFloatSliceEval = JitBulkEval<FloatSliceAssembler>;
