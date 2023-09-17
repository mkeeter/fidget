use crate::{
    eval::types::Grad,
    jit::{JitEval, SimdType},
};

/// Assembler for automatic differentiation / gradient evaluation
pub struct GradSliceAssembler<'a, D>(pub(crate) &'a mut D);
pub type JitGradSliceEval = JitEval<*const Grad>;

// Both x86_64 and AArch64 process 1 gradient per register
impl SimdType for *const Grad {
    const SIMD_SIZE: usize = 1;
}
