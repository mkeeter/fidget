use crate::{
    eval::types::Interval,
    jit::{arch, JitEval},
};
use dynasmrt::VecAssembler;

// TODO: could we use an `Interval` here as well?
pub struct IntervalAssembler<'a>(
    pub(crate) &'a mut VecAssembler<arch::Relocation>,
);
pub type JitIntervalEval = JitEval<Interval>;
