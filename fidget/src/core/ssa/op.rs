/// Opcode for use in an SSA [`Tape`](super::Tape)
#[derive(Copy, Clone, Debug)]
pub enum Op {
    /// Reads one of the inputs (X, Y, Z).  This is the most flexible variable,
    /// and may vary between terms in vector / SIMD evaluation.
    Input(u32, u32),
    /// Represents a variable.  Unlike `Input`, this node is assumed to remain
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_op_size() {
        assert_eq!(std::mem::size_of::<Op>(), 16);
    }
}
