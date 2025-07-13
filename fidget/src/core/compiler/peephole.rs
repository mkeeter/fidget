use super::SsaOp;

/// Peephole optimizer for a tape of [`SsaOp`] instructions
pub struct PeepholeOptimizer {
    /// Normalized SSA tape
    tape: Vec<SsaOp>,
    use_count: Vec<usize>,
}

impl PeepholeOptimizer {
    /// Builds a new peephole optimizer
    pub fn new(mut tape: Vec<SsaOp>) -> Self {
        Self::normalize(&mut tape);
        let mut use_count = vec![0usize; tape.len()];
        for (i, op) in tape.iter().enumerate() {
            if matches!(op, SsaOp::Output(..)) {
                use_count[i] = 1;
            }
            for c in op.iter_children() {
                use_count[c as usize] += 1;
            }
        }

        Self { tape, use_count }
    }

    /// Normalizes a tape such that output index matches tape position
    pub fn normalize(tape: &mut [SsaOp]) {
        Self::normalize_with(tape, tape.len());
    }

    fn normalize_with(tape: &mut [SsaOp], len: usize) {
        let mut remap = vec![None; len];
        for (i, op) in tape.iter_mut().enumerate().rev() {
            if let Some(out) = op.output_mut() {
                assert!(remap[*out as usize].is_none());
                remap[*out as usize] = Some(i as u32);
                *out = i as u32;
            }
            for i in op.iter_children_mut() {
                *i = remap[*i as usize].unwrap();
            }
        }
    }

    pub fn optimize(mut self) -> Vec<SsaOp> {
        let len = self.tape.len();
        while self.peep() {
            // keep going
        }
        self.tape.retain(|op| {
            op.output().is_none_or(|i| self.use_count[i as usize] > 0)
        });
        Self::normalize_with(&mut self.tape, len);
        self.tape
    }

    /// Performs one round of peephole optimization
    fn peep(&mut self) -> bool {
        let mut changed = false;
        for i in 0..self.tape.len() {
            if self.use_count[i] == 0 {
                continue;
            } else if let SsaOp::AddRegImm(out, r, imm) = self.tape[i]
                && let SsaOp::SqrtReg(_, r2) = self.tape[r as usize]
                && let SsaOp::AddRegReg(_, x2, y2) = self.tape[r2 as usize]
                && let SsaOp::SquareReg(_, x) = self.tape[x2 as usize]
                && let SsaOp::SquareReg(_, y) = self.tape[y2 as usize]
            {
                self.use_count[r as usize] -= 1;
                if self.use_count[r as usize] == 0 {
                    self.use_count[r2 as usize] -= 1;
                    if self.use_count[r2 as usize] == 0 {
                        self.use_count[x2 as usize] -= 1;
                        self.use_count[y2 as usize] -= 1;
                    }
                }
                self.tape[i] = SsaOp::RadiusRegRegImm(out, x, y, -imm);
                changed = true;
            } else if let SsaOp::SqrtReg(out, r2) = self.tape[i]
                && let SsaOp::AddRegReg(_, x2, y2) = self.tape[r2 as usize]
                && let SsaOp::SquareReg(_, x) = self.tape[x2 as usize]
                && let SsaOp::SquareReg(_, y) = self.tape[y2 as usize]
            {
                self.use_count[r2 as usize] -= 1;
                if self.use_count[r2 as usize] == 0 {
                    self.use_count[x2 as usize] -= 1;
                    self.use_count[y2 as usize] -= 1;
                }
                self.tape[i] = SsaOp::RadiusRegReg(out, x, y);
                changed = true;
            }
        }
        changed
    }
}
