/// Basic operations that can be performed in a tape
///
/// Arguments, in order, are
/// - Output register
/// - LHS register (or input slot for [`Input`](Op::Input))
/// - RHS register (or immediate for `*Imm`)
///
/// Each "register" may be an SSA slot (represented as a `u32` and never
/// reused), a VM register (represented as a `u8` and reused during evaluation),
/// or physical (e.g. 0-23 for AArch64).
#[derive(Copy, Clone, Debug)]
pub enum TapeOp<T> {
    /// Read one of the inputs (X, Y, Z)
    Input(T, T),

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
    ///
    /// This is never used in SSA tapes, which has infinite registers
    Load(T, u32),

    /// Write from a register to a memory slot
    ///
    /// This is never used in SSA tapes, which has infinite registers
    Store(T, u32),
}

impl<T: Copy> TapeOp<T> {
    /// Returns the output pseudo-register, or `None` for `Store` instructions
    pub fn output(&self) -> Option<T> {
        match self {
            TapeOp::Input(out, ..)
            | TapeOp::Var(out, ..)
            | TapeOp::CopyImm(out, ..)
            | TapeOp::NegReg(out, ..)
            | TapeOp::AbsReg(out, ..)
            | TapeOp::RecipReg(out, ..)
            | TapeOp::SqrtReg(out, ..)
            | TapeOp::SquareReg(out, ..)
            | TapeOp::CopyReg(out, ..)
            | TapeOp::AddRegImm(out, ..)
            | TapeOp::MulRegImm(out, ..)
            | TapeOp::DivRegImm(out, ..)
            | TapeOp::DivImmReg(out, ..)
            | TapeOp::SubImmReg(out, ..)
            | TapeOp::SubRegImm(out, ..)
            | TapeOp::AddRegReg(out, ..)
            | TapeOp::MulRegReg(out, ..)
            | TapeOp::DivRegReg(out, ..)
            | TapeOp::SubRegReg(out, ..)
            | TapeOp::MinRegImm(out, ..)
            | TapeOp::MaxRegImm(out, ..)
            | TapeOp::MinRegReg(out, ..)
            | TapeOp::MaxRegReg(out, ..)
            | TapeOp::Load(out, ..) => Some(*out),
            TapeOp::Store(..) => None,
        }
    }
    /// Returns the number of choices made by the given opcode
    ///
    /// This is always zero or one.
    pub fn choice_count(&self) -> usize {
        match self {
            TapeOp::Input(..)
            | TapeOp::Var(..)
            | TapeOp::CopyImm(..)
            | TapeOp::NegReg(..)
            | TapeOp::AbsReg(..)
            | TapeOp::RecipReg(..)
            | TapeOp::SqrtReg(..)
            | TapeOp::SquareReg(..)
            | TapeOp::CopyReg(..)
            | TapeOp::AddRegImm(..)
            | TapeOp::MulRegImm(..)
            | TapeOp::SubRegImm(..)
            | TapeOp::SubImmReg(..)
            | TapeOp::AddRegReg(..)
            | TapeOp::MulRegReg(..)
            | TapeOp::SubRegReg(..)
            | TapeOp::DivRegReg(..)
            | TapeOp::DivRegImm(..)
            | TapeOp::DivImmReg(..)
            | TapeOp::Load(..)
            | TapeOp::Store(..) => 0,
            TapeOp::MinRegImm(..)
            | TapeOp::MaxRegImm(..)
            | TapeOp::MinRegReg(..)
            | TapeOp::MaxRegReg(..) => 1,
        }
    }
}
