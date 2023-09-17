use crate::jit::{arch, arch::float_slice::SIMD_WIDTH, JitEval, SimdType};
use dynasmrt::VecAssembler;

pub struct FloatSliceAssembler<'a>(
    pub(crate) &'a mut VecAssembler<arch::Relocation>,
);

impl SimdType for *const f32 {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}

pub type JitFloatSliceEval = JitEval<*const f32>;
