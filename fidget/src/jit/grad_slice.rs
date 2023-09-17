use crate::{
    eval::types::Grad,
    jit::{arch, JitEval, SimdType},
};
use dynasmrt::VecAssembler;

/// Assembler for automatic differentiation / gradient evaluation
pub struct GradSliceAssembler<'a>(
    pub(crate) &'a mut VecAssembler<arch::Relocation>,
);
pub type JitGradSliceEval = JitEval<*const Grad>;

// Both x86_64 and AArch64 process 1 gradient per register
impl SimdType for *const Grad {
    const SIMD_SIZE: usize = 1;
}
