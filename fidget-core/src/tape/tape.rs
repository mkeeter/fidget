use crate::{
    asm::{AsmOp, AsmTape},
    eval::Choice,
    tape::SsaTape,
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

    pub fn simplify_with_reg_limit(
        &self,
        choices: &[Choice],
        reg_limit: u8,
    ) -> Self {
        let t = self.0.simplify_with_reg_limit(choices, reg_limit);
        Self(Arc::new(t))
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        let t = self.0.simplify(choices);
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
        self.simplify_with_reg_limit(choices, u8::MAX)
    }

    pub fn simplify_with_reg_limit(
        &self,
        choices: &[Choice],
        reg_limit: u8,
    ) -> Self {
        let (ssa, asm) = self.ssa.simplify(choices, reg_limit);
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

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        eval::{asm::AsmPointEval, point::PointEval},
    };

    #[test]
    fn basic_interpreter() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let one = ctx.constant(1.0);
        let sum = ctx.add(x, one).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = PointEval::<AsmPointEval>::from(tape);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 3.5, 0.0), 3.5);
    }

    #[test]
    fn test_push() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = PointEval::<AsmPointEval>::from(tape.clone());
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = PointEval::<AsmPointEval>::from(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = PointEval::<AsmPointEval>::from(t);
        assert_eq!(eval.eval_p(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.eval_p(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = PointEval::<AsmPointEval>::from(tape.clone());
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = PointEval::<AsmPointEval>::from(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = PointEval::<AsmPointEval>::from(t);
        assert_eq!(eval.eval_p(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.eval_p(3.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn test_ring() {
        let mut ctx = Context::new();
        let c0 = ctx.constant(0.5);
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r = ctx.add(x2, y2).unwrap();
        let c6 = ctx.sub(r, c0).unwrap();
        let c7 = ctx.constant(0.25);
        let c8 = ctx.sub(c7, r).unwrap();
        let c9 = ctx.max(c8, c6).unwrap();

        let tape = ctx.get_tape(c9, u8::MAX);
        assert_eq!(tape.ssa.tape.len(), 8);
    }

    #[test]
    fn test_dupe() {
        use crate::context::Context;
        let mut ctx = Context::new();
        let x = ctx.x();
        let x_squared = ctx.mul(x, x).unwrap();

        let tape = ctx.get_tape(x_squared, u8::MAX);
        assert_eq!(tape.ssa.tape.len(), 2);
    }

    #[test]
    fn test_circle() {
        use crate::context::Context;
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let x_squared = ctx.mul(x, x).unwrap();
        let y_squared = ctx.mul(y, y).unwrap();
        let radius = ctx.add(x_squared, y_squared).unwrap();
        let one = ctx.constant(1.0);
        let circle = ctx.sub(radius, one).unwrap();

        let tape = ctx.get_tape(circle, u8::MAX);
        let mut eval = PointEval::<AsmPointEval>::from(tape);
        assert_eq!(eval.eval_p(0.0, 0.0, 0.0), -1.0);
        assert_eq!(eval.eval_p(1.0, 0.0, 0.0), 0.0);
    }
}
