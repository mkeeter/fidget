use crate::{
    eval::types::Interval,
    jit::{AssemblerData, JitEval},
};

// TODO: could we use an `Interval` here as well?
pub struct IntervalAssembler<'a>(pub(crate) AssemblerData<'a, [f32; 2]>);
pub type JitIntervalEval = JitEval<Interval>;
