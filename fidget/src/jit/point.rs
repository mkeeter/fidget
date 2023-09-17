use crate::jit::JitEval;

pub struct PointAssembler<'a, D>(pub(crate) &'a mut D);
pub type JitPointEval = JitEval<f32>;
