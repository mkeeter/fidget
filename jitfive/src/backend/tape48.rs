use crate::backend::tape32::Choice;

#[derive(Copy, Clone, Debug)]
pub enum ClauseOp48 {
    /// Reads one of the inputs (X, Y, Z)
    Input(u8),

    NegReg(u32),
    AbsReg(u32),
    RecipReg(u32),
    SqrtReg(u32),
    SquareReg(u32),

    AddRegReg(u32, u32),
    MulRegReg(u32, u32),
    SubRegReg(u32, u32),
    MinRegReg(u32, u32),
    MaxRegReg(u32, u32),

    /// Add a register and an immediate
    AddRegImm(u32, f32),
    /// Multiply a register and an immediate
    MulRegImm(u32, f32),
    /// Subtract a register from an immediate
    SubImmReg(u32, f32),
    /// Subtract an immediate from a register
    SubRegImm(u32, f32),
    /// Compute the minimum of a register and an immediate
    MinRegImm(u32, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(u32, f32),

    /// Copy an immediate to a register
    CopyImm(f32),
}

/// Tape storing 48-bit (12-byte) operations:
/// - 4-byte opcode
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs are implicitly stored based on clause index in `tape`, e.g. the
/// first item in the tape writes to slot 0.
#[derive(Clone, Debug)]
pub struct Tape {
    /// Raw instruction tape
    pub tape: Vec<ClauseOp48>,

    /// The number of nodes which store values in the choice array during
    /// interval evaluation.
    pub choice_count: usize,
}

/// Workspace to evaluate a tape
pub struct TapeEval<'a> {
    tape: &'a Tape,
    slots: Vec<f32>,
    choices: Vec<Choice>,
}

impl Tape {
    /// Builds an evaluator which takes a (read-only) reference to this tape
    pub fn get_evaluator(&self) -> TapeEval {
        TapeEval {
            tape: self,
            slots: vec![0.0; self.tape.len()],
            choices: vec![],
        }
    }
}

impl<'a> TapeEval<'a> {
    fn v(&self, i: u32) -> f32 {
        self.slots[i as usize]
    }
    pub fn f(&mut self, x: f32, y: f32) -> f32 {
        for (i, &op) in self.tape.tape.iter().enumerate() {
            self.slots[i] = match op {
                ClauseOp48::Input(i) => match i {
                    0 => x,
                    1 => y,
                    _ => panic!(),
                },
                ClauseOp48::NegReg(i) => -self.v(i),
                ClauseOp48::AbsReg(i) => self.v(i).abs(),
                ClauseOp48::RecipReg(i) => 1.0 / self.v(i),
                ClauseOp48::SqrtReg(i) => self.v(i).sqrt(),
                ClauseOp48::SquareReg(i) => self.v(i) * self.v(i),

                ClauseOp48::AddRegReg(a, b) => self.v(a) + self.v(b),
                ClauseOp48::MulRegReg(a, b) => self.v(a) * self.v(b),
                ClauseOp48::SubRegReg(a, b) => self.v(a) - self.v(b),
                ClauseOp48::MinRegReg(a, b) => self.v(a).min(self.v(b)),
                ClauseOp48::MaxRegReg(a, b) => self.v(a).max(self.v(b)),
                ClauseOp48::AddRegImm(a, imm) => self.v(a) + imm,
                ClauseOp48::MulRegImm(a, imm) => self.v(a) * imm,
                ClauseOp48::SubImmReg(a, imm) => imm - self.v(a),
                ClauseOp48::SubRegImm(a, imm) => self.v(a) - imm,
                ClauseOp48::MinRegImm(a, imm) => self.v(a).min(imm),
                ClauseOp48::MaxRegImm(a, imm) => self.v(a).max(imm),
                ClauseOp48::CopyImm(imm) => imm,
            };
        }
        self.slots[0]
    }
}
