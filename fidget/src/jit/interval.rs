use crate::{eval::types::Interval, jit::JitEval};

// TODO: could we use an `Interval` here as well?
pub struct IntervalAssembler<'a, D>(pub(crate) &'a mut D);
pub type JitIntervalEval = JitEval<Interval>;
