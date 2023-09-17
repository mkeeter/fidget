use crate::jit::{arch, JitEval};
use dynasmrt::VecAssembler;

pub struct PointAssembler<'a>(
    pub(crate) &'a mut VecAssembler<arch::Relocation>,
);
pub type JitPointEval = JitEval<f32>;
