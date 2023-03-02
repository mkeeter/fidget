use crate::jit::{AssemblerData, JitBulkEval, SimdAssembler};

/// Assembler for automatic differentiation / gradient evaluation
pub struct GradSliceAssembler(pub(crate) AssemblerData<[f32; 4]>);

impl SimdAssembler for GradSliceAssembler {
    const SIMD_SIZE: usize = 1;
}

////////////////////////////////////////////////////////////////////////////////

pub type JitGradSliceEval = JitBulkEval<GradSliceAssembler>;
