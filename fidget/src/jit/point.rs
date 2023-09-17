use crate::jit::{AssemblerData, JitEval};

pub struct PointAssembler<'a>(pub(crate) AssemblerData<'a, f32>);
pub type JitPointEval = JitEval<f32>;
