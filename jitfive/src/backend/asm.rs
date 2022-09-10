/// Operation that can be passed directly to the assembler
///
/// This is analagous to the `ClauseOp` family, but assumes relative addressing
/// and a maximum of 256 registers (and hence includes `Load` and `Store`
/// operations).
///
/// Arguments, in order, are
/// - Output register
/// - LHS register (or input slot for `Input`)
/// - RHS register (or immediate for `*Imm`)
#[derive(Copy, Clone, Debug)]
pub enum AsmOp {
    /// Reads one of the inputs (X, Y, Z)
    Input(u8, u8),

    NegReg(u8, u8),
    AbsReg(u8, u8),
    RecipReg(u8, u8),
    SqrtReg(u8, u8),
    SquareReg(u8, u8),

    /// Copies the given register
    CopyReg(u8, u8),

    /// Add a register and an immediate
    AddRegImm(u8, u8, f32),
    /// Multiply a register and an immediate
    MulRegImm(u8, u8, f32),
    /// Subtract a register from an immediate
    SubImmReg(u8, u8, f32),
    /// Subtract an immediate from a register
    SubRegImm(u8, u8, f32),
    /// Compute the minimum of a register and an immediate
    MinRegImm(u8, u8, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(u8, u8, f32),

    AddRegReg(u8, u8, u8),
    MulRegReg(u8, u8, u8),
    SubRegReg(u8, u8, u8),
    MinRegReg(u8, u8, u8),
    MaxRegReg(u8, u8, u8),

    /// Copy an immediate to a register
    CopyImm(u8, f32),
    Load(u8, u32),
    Store(u8, u32),
}

/// Evaluator for a slice of [`AsmOp`]
pub struct AsmEval<'a> {
    /// Instruction tape, in reverse-evaluation order
    pub tape: &'a [AsmOp],
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
