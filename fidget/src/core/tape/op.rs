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
pub enum Op<T = u8> {
    /// Read one of the inputs (X, Y, Z)
    Input(T, u8),

    /// Reads one of the variables
    Var(T, u32),

    /// Negate the given register
    NegReg(T, T),

    /// Take the absolute value of the given register
    AbsReg(T, T),

    /// Take the reciprocal of the given register (1.0 / value)
    RecipReg(T, T),

    /// Take the square root of the given register
    SqrtReg(T, T),

    /// Square the given register
    SquareReg(T, T),

    /// Copies the given register
    CopyReg(T, T),

    /// Add a register and an immediate
    AddRegImm(T, T, f32),
    /// Multiply a register and an immediate
    MulRegImm(T, T, f32),
    /// Divides a register and an immediate
    DivRegImm(T, T, f32),
    /// Divides an immediate by a register
    DivImmReg(T, T, f32),
    /// Subtract a register from an immediate
    SubImmReg(T, T, f32),
    /// Subtract an immediate from a register
    SubRegImm(T, T, f32),
    /// Compute the minimum of a register and an immediate
    MinRegImm(T, T, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(T, T, f32),

    /// Add two registers
    AddRegReg(T, T, T),
    /// Multiply two registers
    MulRegReg(T, T, T),
    /// Divides two registers
    DivRegReg(T, T, T),
    /// Subtract one register from another
    SubRegReg(T, T, T),
    /// Take the minimum of two registers
    MinRegReg(T, T, T),
    /// Take the maximum of two registers
    MaxRegReg(T, T, T),

    /// Copy an immediate to a register
    CopyImm(T, f32),

    /// Read from a memory slot to a register
    Load(T, u32),
    /// Write from a register to a memory slot
    Store(T, u32),
}

impl<T: Copy> Op<T> {
    pub fn output(&self) -> Option<T> {
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
