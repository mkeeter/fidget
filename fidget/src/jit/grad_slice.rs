use crate::{
    eval::types::Grad,
    jit::{AssemblerData, SimdSize},
};

/// Assembler for automatic differentiation / gradient evaluation
pub struct GradSliceAssembler(pub(crate) AssemblerData<Grad>);

// Both x86_64 and AArch64 process 1 gradient per register
impl SimdSize for Grad {
    const SIMD_SIZE: usize = 1;
}
