use crate::{
    asm::{AsmOp, AsmTape},
    eval::Choice,
    tape::{SsaTape, Workspace},
};
use std::sync::Arc;

/// Light-weight handle for tape data
///
/// This can be passed by value and cloned.
#[derive(Clone)]
pub struct Tape(Arc<TapeData>);
impl Tape {
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let t = TapeData::from_ssa(ssa, reg_limit);
        Self(Arc::new(t))
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        let t = self.0.simplify(choices);
        Self(Arc::new(t))
    }

    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
    ) -> Self {
        let t = self.0.simplify_with(choices, workspace);
        Self(Arc::new(t))
    }
}

impl std::ops::Deref for Tape {
    type Target = TapeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`Tape`](Self) stores two different representations:
/// - A tape in SSA form, suitable for use during tape simplification
/// - A [`Vec<AsmOp>`](crate::asm::AsmOp), ready to be fed into an assembler,
///   (e.g. [`dynasm`](crate::asm::dynasm)).
///
/// We keep both because SSA form makes tape shortening easier, while the `asm`
/// data already has registers assigned for lowering into machine assembly.
pub struct TapeData {
    ssa: SsaTape,
    asm: AsmTape,
}

impl TapeData {
    /// Returns the length of the internal `AsmOp` tape
    pub fn len(&self) -> usize {
        self.asm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.asm.is_empty()
    }

    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocated spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.ssa.choice_count
    }
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let asm = ssa.get_asm(reg_limit);
        Self { ssa, asm }
    }

    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        let (ssa, asm) = self.ssa.simplify(choices, self.asm.reg_limit());
        Self { ssa, asm }
    }

    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
    ) -> Self {
        let (ssa, asm) =
            self.ssa
                .simplify_with(choices, self.asm.reg_limit(), workspace);
        Self { ssa, asm }
    }

    /// Produces an iterator that visits [`AsmOp`](crate::asm::AsmOp) values in
    /// evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = AsmOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    pub fn pretty_print(&self) {
        self.ssa.pretty_print()
    }
}
