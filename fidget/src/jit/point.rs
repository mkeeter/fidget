use crate::jit::{AssemblerData, JitTracingEval};

pub struct PointAssembler(pub(crate) AssemblerData<f32>);
pub type JitPointEval = JitTracingEval<PointAssembler>;
