use crate::{eval::types::Interval, jit::JitEval};

pub struct IntervalAssembler;
pub type JitIntervalEval = JitEval<Interval>;
