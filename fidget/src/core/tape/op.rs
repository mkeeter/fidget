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

impl TapeOp {
    /// Returns the number of data fields associated with this opcode
    pub fn data_count(&self) -> usize {
        match self {
            TapeOp::Input
            | TapeOp::CopyImm
            | TapeOp::NegReg
            | TapeOp::AbsReg
            | TapeOp::RecipReg
            | TapeOp::SqrtReg
            | TapeOp::SquareReg
            | TapeOp::CopyReg => 1,

            TapeOp::AddRegImm
            | TapeOp::MulRegImm
            | TapeOp::SubRegImm
            | TapeOp::SubImmReg
            | TapeOp::AddRegReg
            | TapeOp::MulRegReg
            | TapeOp::SubRegReg
            | TapeOp::DivRegReg
            | TapeOp::DivRegImm
            | TapeOp::DivImmReg
            | TapeOp::MinRegImm
            | TapeOp::MaxRegImm
            | TapeOp::MinRegReg
            | TapeOp::MaxRegReg => 2,
        }
    }
    pub fn choice_count(&self) -> usize {
        match self {
            TapeOp::Input
            | TapeOp::CopyImm
            | TapeOp::NegReg
            | TapeOp::AbsReg
            | TapeOp::RecipReg
            | TapeOp::SqrtReg
            | TapeOp::SquareReg
            | TapeOp::CopyReg
            | TapeOp::AddRegImm
            | TapeOp::MulRegImm
            | TapeOp::SubRegImm
            | TapeOp::SubImmReg
            | TapeOp::AddRegReg
            | TapeOp::MulRegReg
            | TapeOp::SubRegReg
            | TapeOp::DivRegReg
            | TapeOp::DivRegImm
            | TapeOp::DivImmReg => 0,
            TapeOp::MinRegImm
            | TapeOp::MaxRegImm
            | TapeOp::MinRegReg
            | TapeOp::MaxRegReg => 1,
        }
    }
}
