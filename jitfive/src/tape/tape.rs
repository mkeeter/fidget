use crate::{
    asm::{AsmEval, AsmOp, Choice},
    tape::SsaTape,
};

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`Tape`](Self) stores two different representations:
/// - An [`SsaTape`](crate::tape::SsaTape), suitable for use during tape
///   simplification
/// - A [`Vec<AsmOp>`](crate::asm::AsmOp), ready to be fed into an assembler,
///   (e.g. [`dynasm`](crate::asm::dynasm)).
///
/// We keep both because SSA form makes tape shortening easier, while the `asm`
/// data already has registers assigned.
pub struct Tape {
    ssa: SsaTape,
    asm: Vec<AsmOp>,
    reg_limit: u8,
}

impl Tape {
    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required if pre-allocating space for evaluation that writes
    /// [`Choice`](crate::asm::Choice) values.
    pub fn choice_count(&self) -> usize {
        self.ssa.choice_count
    }
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let asm = ssa.get_asm(reg_limit);
        Self {
            ssa,
            asm,
            reg_limit,
        }
    }

    pub fn get_evaluator(&self) -> AsmEval {
        AsmEval::new(&self.asm)
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        let (ssa, asm) = self.ssa.simplify(choices, self.reg_limit);
        Self {
            ssa,
            asm,
            reg_limit: self.reg_limit,
        }
    }

    /// Produces an iterator that visits [`AsmOp`](crate::asm::AsmOp) values in
    /// evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = AsmOp> + '_ {
        self.asm.iter().cloned().rev()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    #[test]
    fn basic_interpreter() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let one = ctx.constant(1.0);
        let sum = ctx.add(x, one).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 3.5, 0.0), 3.5);
    }

    #[test]
    fn test_push() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);
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
}
