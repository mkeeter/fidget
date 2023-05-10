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
    /// Returns the output register associated with this operation
    ///
    /// Every operation has an output register except [`Store`](Op::Store)
    /// (which takes a single input register and writes it to memory)
    pub fn out_reg(&self) -> Option<u8> {
        match self {
            Op::Load(..) => None,
            Op::Store(reg, ..)
            | Op::Var(reg, ..)
            | Op::Input(reg, ..)
            | Op::CopyImm(reg, ..)
            | Op::CopyReg(reg, ..)
            | Op::AddRegReg(reg, ..)
            | Op::AddRegImm(reg, ..)
            | Op::NegReg(reg, ..)
            | Op::AbsReg(reg, ..)
            | Op::RecipReg(reg, ..)
            | Op::SqrtReg(reg, ..)
            | Op::SquareReg(reg, ..)
            | Op::MulRegReg(reg, ..)
            | Op::MulRegImm(reg, ..)
            | Op::MinRegReg(reg, ..)
            | Op::MinRegImm(reg, ..)
            | Op::MaxRegReg(reg, ..)
            | Op::MaxRegImm(reg, ..)
            | Op::DivRegReg(reg, ..)
            | Op::DivRegImm(reg, ..)
            | Op::DivImmReg(reg, ..)
            | Op::SubRegReg(reg, ..)
            | Op::SubRegImm(reg, ..)
            | Op::SubImmReg(reg, ..) => Some(*reg),
        }
    }

    /// Returns an iterator over input registers
    pub fn input_reg_iter(&self) -> impl Iterator<Item = u8> {
        match *self {
            Op::Input(..)
            | Op::Var(..)
            | Op::CopyReg(..)
            | Op::CopyImm(..)
            | Op::Load(..) => [None, None],
            Op::NegReg(_out, arg)
            | Op::AbsReg(_out, arg)
            | Op::RecipReg(_out, arg)
            | Op::SqrtReg(_out, arg)
            | Op::SquareReg(_out, arg)
            | Op::AddRegImm(_out, arg, ..)
            | Op::MulRegImm(_out, arg, ..)
            | Op::DivRegImm(_out, arg, ..)
            | Op::DivImmReg(_out, arg, ..)
            | Op::SubImmReg(_out, arg, ..)
            | Op::SubRegImm(_out, arg, ..)
            | Op::MinRegImm(_out, arg, ..)
            | Op::MaxRegImm(_out, arg, ..) => [Some(arg), None],
            Op::AddRegReg(_out, lhs, rhs)
            | Op::MulRegReg(_out, lhs, rhs)
            | Op::DivRegReg(_out, lhs, rhs)
            | Op::SubRegReg(_out, lhs, rhs)
            | Op::MinRegReg(_out, lhs, rhs)
            | Op::MaxRegReg(_out, lhs, rhs) => [Some(lhs), Some(rhs)],
            Op::Store(reg, _mem) => [Some(reg), None],
        }
        .into_iter()
        .flatten()
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
