/// Opcode for use in a [`Tape`](super::Tape)
#[derive(Copy, Clone, Debug)]
pub enum TapeOp {
    /// Reads one of the inputs (X, Y, Z)
    Input,
    /// Copy an immediate to a register
    CopyImm,

    /// Negates a register
    NegReg,
    /// Takes the absolute value of a register
    AbsReg,
    /// Takes the reciprocal of a register
    RecipReg,
    /// Takes the square root of a register
    SqrtReg,
    /// Squares a register
    SquareReg,

    /// Copies the given register
    CopyReg,

    /// Add a register and an immediate
    AddRegImm,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Divides a register and an immediate
    DivRegImm,
    /// Divides an immediate by a register
    DivImmReg,
    /// Subtract a register from an immediate
    SubImmReg,
    /// Subtract an immediate from a register
    SubRegImm,

    /// Adds two registers
    AddRegReg,
    /// Multiplies two registers
    MulRegReg,
    /// Divides two registers
    DivRegReg,
    /// Subtracts two registers
    SubRegReg,

    /// Compute the minimum of a register and an immediate
    MinRegImm,
    /// Compute the maximum of a register and an immediate
    MaxRegImm,
    /// Compute the minimum of two registers
    MinRegReg,
    /// Compute the maximum of two registers
    MaxRegReg,
}
