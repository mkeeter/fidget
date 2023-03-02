use crate::jit::{AssemblerData, JitTracingEval};

pub struct IntervalAssembler(pub(crate) AssemblerData<[f32; 2]>);
pub type JitIntervalEval = JitTracingEval<IntervalAssembler>;
