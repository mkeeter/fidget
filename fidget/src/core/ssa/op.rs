/// Opcode for use in an SSA [`Tape`](super::Tape)
#[derive(Copy, Clone, Debug)]
pub enum Op {
    /// Reads one of the inputs (X, Y, Z).  This is the most flexible variable,
    /// and may vary between terms in vector / SIMD evaluation.
    Input(u32, u32),
    /// A single variable.  Unlike `Input`, this node is assumed to remain
    /// constant across all terms in vector / SIMD evaluation, but may be edited
    /// by the user in between evaluation.
    Var(u32, u32),
    /// Copy an immediate to a register
    CopyImm(u32, f32),

    /// Negates a register
    NegReg(u32, u32),
    /// Takes the absolute value of a register
    AbsReg(u32, u32),
    /// Takes the reciprocal of a register
    RecipReg(u32, u32),
    /// Takes the square root of a register
    SqrtReg(u32, u32),
    /// Squares a register
    SquareReg(u32, u32),

    /// Copies the given register
    CopyReg(u32, u32),

    /// Add a register and an immediate
    AddRegImm(u32, u32, f32),
    /// Multiply a register and an immediate
    MulRegImm(u32, u32, f32),
    /// Divides a register and an immediate
    DivRegImm(u32, u32, f32),
    /// Divides an immediate by a register
    DivImmReg(u32, u32, f32),
    /// Subtract a register from an immediate
    SubImmReg(u32, u32, f32),
    /// Subtract an immediate from a register
    SubRegImm(u32, u32, f32),

    /// Adds two registers
    AddRegReg(u32, u32, u32),
    /// Multiplies two registers
    MulRegReg(u32, u32, u32),
    /// Divides two registers
    DivRegReg(u32, u32, u32),
    /// Subtracts two registers
    SubRegReg(u32, u32, u32),

    /// Compute the minimum of a register and an immediate
    MinRegImm(u32, u32, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(u32, u32, f32),
    /// Compute the minimum of two registers
    MinRegReg(u32, u32, u32),
    /// Compute the maximum of two registers
    MaxRegReg(u32, u32, u32),
}

impl Op {
    /// Returns the index of the output variable
    pub fn output(&self) -> u32 {
        match self {
            Op::Input(out, ..)
            | Op::Var(out, ..)
            | Op::CopyImm(out, ..)
            | Op::NegReg(out, ..)
            | Op::AbsReg(out, ..)
            | Op::RecipReg(out, ..)
            | Op::SqrtReg(out, ..)
            | Op::SquareReg(out, ..)
            | Op::CopyReg(out, ..)
            | Op::AddRegImm(out, ..)
            | Op::MulRegImm(out, ..)
            | Op::DivRegImm(out, ..)
            | Op::DivImmReg(out, ..)
            | Op::SubImmReg(out, ..)
            | Op::SubRegImm(out, ..)
            | Op::AddRegReg(out, ..)
            | Op::MulRegReg(out, ..)
            | Op::DivRegReg(out, ..)
            | Op::SubRegReg(out, ..)
            | Op::MinRegImm(out, ..)
            | Op::MaxRegImm(out, ..)
            | Op::MinRegReg(out, ..)
            | Op::MaxRegReg(out, ..) => *out,
        }
    }
    /// Returns the number of choices made by the given opcode
    ///
    /// This is always zero or one.
    pub fn choice_count(&self) -> usize {
        match self {
            Op::Input(..)
            | Op::Var(..)
            | Op::CopyImm(..)
            | Op::NegReg(..)
            | Op::AbsReg(..)
            | Op::RecipReg(..)
            | Op::SqrtReg(..)
            | Op::SquareReg(..)
            | Op::CopyReg(..)
            | Op::AddRegImm(..)
            | Op::MulRegImm(..)
            | Op::SubRegImm(..)
            | Op::SubImmReg(..)
            | Op::AddRegReg(..)
            | Op::MulRegReg(..)
            | Op::SubRegReg(..)
            | Op::DivRegReg(..)
            | Op::DivRegImm(..)
            | Op::DivImmReg(..) => 0,
            Op::MinRegImm(..)
            | Op::MaxRegImm(..)
            | Op::MinRegReg(..)
            | Op::MaxRegReg(..) => 1,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_op_size() {
        assert_eq!(std::mem::size_of::<Op>(), 16);
    }
}
