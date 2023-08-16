use crate::jit::{AssemblerData, JitEval};

pub struct PointAssembler(pub(crate) AssemblerData<f32>);
pub type JitPointEval = JitEval<f32>;
