use crate::asm::AsmOp;

/// Evaluator for a slice of [`AsmOp`]
pub struct AsmEval<'a> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'a [AsmOp],
    /// Workspace for data
    slots: Vec<f32>,
}

impl<'a> AsmEval<'a> {
    pub fn new(tape: &'a [AsmOp]) -> Self {
        Self {
            tape,
            slots: vec![],
        }
    }
    fn v(&mut self, i: u8) -> &mut f32 {
        if i as usize >= self.slots.len() {
            self.slots.resize(i as usize + 1, std::f32::NAN);
        }
        &mut self.slots[i as usize]
    }
    pub fn f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        for &op in self.tape.iter().rev() {
            use AsmOp::*;
            match op {
                Input(out, i) => {
                    *self.v(out) = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                NegReg(out, arg) => {
                    *self.v(out) = -*self.v(arg);
                }
                AbsReg(out, arg) => {
                    *self.v(out) = self.v(arg).abs();
                }
                RecipReg(out, arg) => {
                    *self.v(out) = 1.0 / *self.v(arg);
                }
                SqrtReg(out, arg) => {
                    *self.v(out) = self.v(arg).sqrt();
                }
                SquareReg(out, arg) => {
                    *self.v(out) = *self.v(arg) * *self.v(arg)
                }
                CopyReg(out, arg) => *self.v(out) = *self.v(arg),
                AddRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) + imm;
                }
                MulRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) * imm;
                }
                SubImmReg(out, arg, imm) => {
                    *self.v(out) = imm - *self.v(arg);
                }
                SubRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) - imm;
                }
                MinRegImm(out, arg, imm) => {
                    *self.v(out) = self.v(arg).min(imm);
                }
                MaxRegImm(out, arg, imm) => {
                    *self.v(out) = self.v(arg).max(imm);
                }
                AddRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) + *self.v(rhs)
                }
                MulRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) * *self.v(rhs)
                }
                SubRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) - *self.v(rhs)
                }
                MinRegReg(out, lhs, rhs) => {
                    *self.v(out) = self.v(lhs).min(*self.v(rhs))
                }
                MaxRegReg(out, lhs, rhs) => {
                    *self.v(out) = self.v(lhs).max(*self.v(rhs))
                }
                CopyImm(out, imm) => {
                    *self.v(out) = imm;
                }
                Load(out, mem) => {
                    *self.v(out) = self.slots[mem as usize];
                }
                Store(out, mem) => {
                    if mem as usize >= self.slots.len() {
                        self.slots.resize(mem as usize + 1, std::f32::NAN);
                    }
                    self.slots[mem as usize] = *self.v(out);
                }
            }
        }
        self.slots[0]
    }
}
