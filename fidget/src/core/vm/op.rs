/// Operation that can be passed directly to the assembler
///
/// Note that these operations **are not** in SSA form; they operate on a
/// limited number of registers and will reuse them when needed.
///
/// Arguments, in order, are
/// - Output register
/// - LHS register (or input slot for [`Input`](Op::Input))
/// - RHS register (or immediate for `*Imm`)
#[derive(Copy, Clone, Debug)]
pub enum Op {
    /// Read one of the inputs (X, Y, Z)
    Input(u8, u8),

    /// Reads one of the variables
    Var(u8, u32),

    /// Negate the given register
    NegReg(u8, u8),

    /// Take the absolute value of the given register
    AbsReg(u8, u8),

    /// Take the reciprocal of the given register (1.0 / value)
    RecipReg(u8, u8),

    /// Take the square root of the given register
    SqrtReg(u8, u8),

    /// Square the given register
    SquareReg(u8, u8),

    /// Copies the given register
    CopyReg(u8, u8),

    /// Add a register and an immediate
    AddRegImm(u8, u8, f32),
    /// Multiply a register and an immediate
    MulRegImm(u8, u8, f32),
    /// Divides a register and an immediate
    DivRegImm(u8, u8, f32),
    /// Divides an immediate by a register
    DivImmReg(u8, u8, f32),
    /// Subtract a register from an immediate
    SubImmReg(u8, u8, f32),
    /// Subtract an immediate from a register
    SubRegImm(u8, u8, f32),
    /// Compute the minimum of a register and an immediate
    MinRegImm(u8, u8, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(u8, u8, f32),

    /// Add two registers
    AddRegReg(u8, u8, u8),
    /// Multiply two registers
    MulRegReg(u8, u8, u8),
    /// Divides two registers
    DivRegReg(u8, u8, u8),
    /// Subtract one register from another
    SubRegReg(u8, u8, u8),
    /// Take the minimum of two registers
    MinRegReg(u8, u8, u8),
    /// Take the maximum of two registers
    MaxRegReg(u8, u8, u8),

    /// Copy an immediate to a register
    CopyImm(u8, f32),

    /// Read from a memory slot to a register
    Load(u8, u32),
    /// Write from a register to a memory slot
    Store(u8, u32),
}

impl Op {
    pub fn output(&self) -> Option<u8> {
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
            | Op::MaxRegReg(out, ..) => Some(*out),
            Op::Load(..) | Op::Store(..) => None,
        }
    }
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

            Op::Load(..) | Op::Store(..) => 0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_vm_op_size() {
        assert_eq!(std::mem::size_of::<Op>(), 8);
    }
}
