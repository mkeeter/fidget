use super::ChoiceIndex;

/// Operation that can be passed directly to the assembler
///
/// Note that these operations **are not** in SSA form; they operate on a
/// limited number of registers and will reuse them when needed.
///
/// Arguments, by name, are
/// - `out`: Output register
/// - `lhs`: LHS register
/// - `rhs`: RHS register
/// - `imm`: immediate value
/// - `input`: input index (0, 1, 2 for X, Y, Z)
/// - `var`: variable index
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug)]
pub enum Op {
    /// Read one of the inputs (X, Y, Z)
    Input { out: u8, input: u8 },

    /// Reads one of the variables
    Var { out: u8, var: u32 },

    /// Negate the given register
    NegReg { out: u8, arg: u8 },

    /// Take the absolute value of the given register
    AbsReg { out: u8, arg: u8 },

    /// Take the reciprocal of the given register (1.0 / value)
    RecipReg { out: u8, arg: u8 },

    /// Take the square root of the given register
    SqrtReg { out: u8, arg: u8 },

    /// Square the given register
    SquareReg { out: u8, arg: u8 },

    /// Add a register and an immediate
    AddRegImm { out: u8, arg: u8, imm: f32 },
    /// Multiply a register and an immediate
    MulRegImm { out: u8, arg: u8, imm: f32 },
    /// Divides a register and an immediate
    DivRegImm { out: u8, arg: u8, imm: f32 },
    /// Divides an immediate by a register
    DivImmReg { out: u8, arg: u8, imm: f32 },
    /// Subtract a register from an immediate
    SubImmReg { out: u8, arg: u8, imm: f32 },
    /// Subtract an immediate from a register
    SubRegImm { out: u8, arg: u8, imm: f32 },

    /// Compute the minimum of a register and an immediate
    MinRegImm { out: u8, arg: u8, imm: f32 },
    /// Compute the maximum of a register and an immediate
    MaxRegImm { out: u8, arg: u8, imm: f32 },

    /// Compute the in-place minimum of a memory slot and an immediate
    MinMemImmChoice {
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    },
    /// Compute the in-place maximum of a memory slot and an immediate
    MaxMemImmChoice {
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    },
    /// Compute the in-place minimum of a register and an immediate
    MinRegImmChoice {
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    },
    /// Compute the in-place maximum of a register and an immediate
    MaxRegImmChoice {
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    },

    /// Add two registers
    AddRegReg { out: u8, lhs: u8, rhs: u8 },
    /// Multiply two registers
    MulRegReg { out: u8, lhs: u8, rhs: u8 },
    /// Divides two registers
    DivRegReg { out: u8, lhs: u8, rhs: u8 },
    /// Subtract one register from another
    SubRegReg { out: u8, lhs: u8, rhs: u8 },
    /// Take the minimum of two registers
    MinRegReg { out: u8, lhs: u8, rhs: u8 },
    /// Take the maximum of two registers
    MaxRegReg { out: u8, lhs: u8, rhs: u8 },

    /// Compute the in-place minimum of a memory slot and a register
    MinMemRegChoice {
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    },
    /// Compute the in-place maximum of a memory slot and a register
    MaxMemRegChoice {
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    },
    /// Compute the in-place minimum of two registers
    MinRegRegChoice {
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    },
    /// Compute the in-place maximum of two registers
    MaxRegRegChoice {
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    },

    /// Copies the given immediate to a register and sets the choice bit
    CopyImmRegChoice {
        out: u8,
        imm: f32,
        choice: ChoiceIndex,
    },

    /// Copies the given immediate to a memory slot and sets the choice bit
    CopyImmMemChoice {
        out: u32,
        imm: f32,
        choice: ChoiceIndex,
    },

    /// Copies the given register to a memory slot and sets the choice bit
    CopyRegRegChoice {
        out: u8,
        arg: u8,
        choice: ChoiceIndex,
    },

    /// Copies the given register to a memory slot and sets the choice bit
    CopyRegMemChoice {
        out: u32,
        arg: u8,
        choice: ChoiceIndex,
    },

    /// Copy an immediate to a register
    CopyImm { out: u8, imm: f32 },

    /// Copy from one register to another
    CopyReg { out: u8, arg: u8 },

    /// Read from a memory slot to a register
    Load { reg: u8, mem: u32 },
    /// Write from a register to a memory slot
    Store { reg: u8, mem: u32 },
}

impl Op {
    /// Returns the output register associated with this operation
    ///
    /// Every operation has an output register except [`Store`](Op::Store)
    /// (which takes a single input register and writes it to memory) and the
    /// `(Min|Max)Mem(Reg|Imm)Choice` family of operations, which operate
    /// in-place on a memory slot.
    pub fn out_reg(&self) -> Option<u8> {
        match self {
            Op::Load { .. }
            | Op::MinMemRegChoice { .. }
            | Op::MinMemImmChoice { .. }
            | Op::MaxMemRegChoice { .. }
            | Op::MaxMemImmChoice { .. }
            | Op::CopyImmMemChoice { .. }
            | Op::CopyRegMemChoice { .. } => None,
            Op::Store { reg: out, .. }
            | Op::Var { out, .. }
            | Op::Input { out, .. }
            | Op::CopyImm { out, .. }
            | Op::AddRegReg { out, .. }
            | Op::AddRegImm { out, .. }
            | Op::NegReg { out, .. }
            | Op::AbsReg { out, .. }
            | Op::RecipReg { out, .. }
            | Op::SqrtReg { out, .. }
            | Op::SquareReg { out, .. }
            | Op::MulRegReg { out, .. }
            | Op::MulRegImm { out, .. }
            | Op::MinRegReg { out, .. }
            | Op::MinRegImm { out, .. }
            | Op::MaxRegReg { out, .. }
            | Op::MaxRegImm { out, .. }
            | Op::DivRegReg { out, .. }
            | Op::DivRegImm { out, .. }
            | Op::DivImmReg { out, .. }
            | Op::SubRegReg { out, .. }
            | Op::SubRegImm { out, .. }
            | Op::SubImmReg { out, .. }
            | Op::CopyReg { out, .. } => Some(*out),
            Op::MinRegRegChoice { reg, .. }
            | Op::MaxRegRegChoice { reg, .. }
            | Op::MinRegImmChoice { reg, .. }
            | Op::MaxRegImmChoice { reg, .. }
            | Op::CopyImmRegChoice { out: reg, .. }
            | Op::CopyRegRegChoice { out: reg, .. } => Some(*reg),
        }
    }

    /// Returns an iterator over input registers
    ///
    /// Note that `inout` registers (e.g. `reg` in `MinRegRegChoice`) do not
    /// count as input registers for the purpose of this function!
    pub fn input_reg_iter(&self) -> impl Iterator<Item = u8> {
        match *self {
            Op::Input { .. }
            | Op::Var { .. }
            | Op::CopyImm { .. }
            | Op::Load { .. }
            | Op::MinMemImmChoice { .. }
            | Op::MaxMemImmChoice { .. }
            | Op::CopyImmMemChoice { .. }
            | Op::CopyImmRegChoice { .. }
            | Op::MinRegImmChoice { .. }
            | Op::MaxRegImmChoice { .. } => [None, None],
            Op::NegReg { arg, .. }
            | Op::AbsReg { arg, .. }
            | Op::RecipReg { arg, .. }
            | Op::SqrtReg { arg, .. }
            | Op::SquareReg { arg, .. }
            | Op::AddRegImm { arg, .. }
            | Op::MulRegImm { arg, .. }
            | Op::DivRegImm { arg, .. }
            | Op::DivImmReg { arg, .. }
            | Op::SubImmReg { arg, .. }
            | Op::SubRegImm { arg, .. }
            | Op::MinRegImm { arg, .. }
            | Op::MaxRegImm { arg, .. }
            | Op::MinMemRegChoice { arg, .. }
            | Op::MaxMemRegChoice { arg, .. }
            | Op::CopyReg { arg, .. }
            | Op::CopyRegRegChoice { arg, .. }
            | Op::CopyRegMemChoice { arg, .. }
            | Op::MinRegRegChoice { arg, .. }
            | Op::MaxRegRegChoice { arg, .. } => [Some(arg), None],
            Op::AddRegReg { lhs, rhs, .. }
            | Op::MulRegReg { lhs, rhs, .. }
            | Op::DivRegReg { lhs, rhs, .. }
            | Op::SubRegReg { lhs, rhs, .. }
            | Op::MinRegReg { lhs, rhs, .. }
            | Op::MaxRegReg { lhs, rhs, .. } => [Some(lhs), Some(rhs)],
            Op::Store { reg, .. } => [Some(reg), None],
        }
        .into_iter()
        .flatten()
    }

    /// Returns a mutable iterator over all registers (input and output)
    pub fn reg_iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        match self {
            Op::MinMemImmChoice { .. }
            | Op::MaxMemImmChoice { .. }
            | Op::CopyImmMemChoice { .. } => [None, None, None],
            Op::Input { out, .. }
            | Op::Var { out, .. }
            | Op::CopyImm { out, .. }
            | Op::CopyImmRegChoice { out, .. } => [Some(out), None, None],
            Op::Load { reg, .. } => [Some(reg), None, None],
            Op::MinMemRegChoice { arg, .. }
            | Op::MaxMemRegChoice { arg, .. }
            | Op::CopyRegMemChoice { arg, .. } => [Some(arg), None, None],
            Op::NegReg { out, arg, .. }
            | Op::AbsReg { out, arg, .. }
            | Op::RecipReg { out, arg, .. }
            | Op::SqrtReg { out, arg, .. }
            | Op::SquareReg { out, arg, .. }
            | Op::AddRegImm { out, arg, .. }
            | Op::MulRegImm { out, arg, .. }
            | Op::DivRegImm { out, arg, .. }
            | Op::DivImmReg { out, arg, .. }
            | Op::SubImmReg { out, arg, .. }
            | Op::SubRegImm { out, arg, .. }
            | Op::MinRegImm { out, arg, .. }
            | Op::MaxRegImm { out, arg, .. }
            | Op::CopyReg { arg, out }
            | Op::CopyRegRegChoice { arg, out, .. } => {
                [Some(out), Some(arg), None]
            }
            Op::AddRegReg { out, lhs, rhs }
            | Op::MulRegReg { out, lhs, rhs }
            | Op::DivRegReg { out, lhs, rhs }
            | Op::SubRegReg { out, lhs, rhs }
            | Op::MinRegReg { out, lhs, rhs }
            | Op::MaxRegReg { out, lhs, rhs } => {
                [Some(out), Some(lhs), Some(rhs)]
            }
            Op::Store { reg, .. } => [Some(reg), None, None],
            Op::MinRegRegChoice { reg, arg, .. }
            | Op::MaxRegRegChoice { reg, arg, .. } => {
                [Some(reg), Some(arg), None]
            }
            Op::MinRegImmChoice { reg, .. }
            | Op::MaxRegImmChoice { reg, .. } => [Some(reg), None, None],
        }
        .into_iter()
        .flatten()
    }

    /// Returns a register used for an in-place instruction, or `None`
    pub fn inout_reg(&self) -> Option<u8> {
        match *self {
            Op::Input { .. }
            | Op::Var { .. }
            | Op::CopyImm { .. }
            | Op::Load { .. }
            | Op::MinMemImmChoice { .. }
            | Op::MaxMemImmChoice { .. }
            | Op::CopyImmMemChoice { .. }
            | Op::CopyImmRegChoice { .. }
            | Op::NegReg { .. }
            | Op::AbsReg { .. }
            | Op::RecipReg { .. }
            | Op::SqrtReg { .. }
            | Op::SquareReg { .. }
            | Op::AddRegImm { .. }
            | Op::MulRegImm { .. }
            | Op::DivRegImm { .. }
            | Op::DivImmReg { .. }
            | Op::SubImmReg { .. }
            | Op::SubRegImm { .. }
            | Op::MinRegImm { .. }
            | Op::MaxRegImm { .. }
            | Op::MinMemRegChoice { .. }
            | Op::MaxMemRegChoice { .. }
            | Op::CopyReg { .. }
            | Op::CopyRegRegChoice { .. }
            | Op::CopyRegMemChoice { .. }
            | Op::AddRegReg { .. }
            | Op::MulRegReg { .. }
            | Op::DivRegReg { .. }
            | Op::SubRegReg { .. }
            | Op::MinRegReg { .. }
            | Op::MaxRegReg { .. }
            | Op::Store { .. } => None,

            Op::MinRegImmChoice { reg, .. }
            | Op::MaxRegImmChoice { reg, .. }
            | Op::MinRegRegChoice { reg, .. }
            | Op::MaxRegRegChoice { reg, .. } => Some(reg),
        }
    }

    /// Returns an iterator over all slots
    ///
    /// This includes both inputs and output, and both registers and memory.
    pub fn iter_slots(&self) -> impl Iterator<Item = u32> {
        match *self {
            Op::Input { out, .. }
            | Op::Var { out, .. }
            | Op::CopyImm { out, .. }
            | Op::CopyImmRegChoice { out, .. } => {
                [Some(out as u32), None, None]
            }
            Op::NegReg { out, arg }
            | Op::AbsReg { out, arg }
            | Op::RecipReg { out, arg }
            | Op::SqrtReg { out, arg }
            | Op::SquareReg { out, arg }
            | Op::AddRegImm { out, arg, .. }
            | Op::MulRegImm { out, arg, .. }
            | Op::DivRegImm { out, arg, .. }
            | Op::DivImmReg { out, arg, .. }
            | Op::SubImmReg { out, arg, .. }
            | Op::SubRegImm { out, arg, .. }
            | Op::MinRegImm { out, arg, .. }
            | Op::MaxRegImm { out, arg, .. }
            | Op::CopyReg { out, arg }
            | Op::CopyRegRegChoice { out, arg, .. } => {
                [Some(out as u32), Some(arg as u32), None]
            }
            Op::MinMemImmChoice { mem, .. }
            | Op::MaxMemImmChoice { mem, .. }
            | Op::CopyImmMemChoice { out: mem, .. } => [Some(mem), None, None],
            Op::AddRegReg { out, lhs, rhs }
            | Op::MulRegReg { out, lhs, rhs }
            | Op::DivRegReg { out, lhs, rhs }
            | Op::SubRegReg { out, lhs, rhs }
            | Op::MinRegReg { out, lhs, rhs }
            | Op::MaxRegReg { out, lhs, rhs } => {
                [Some(out as u32), Some(lhs as u32), Some(rhs as u32)]
            }
            Op::MinMemRegChoice { mem, arg, .. }
            | Op::MaxMemRegChoice { mem, arg, .. } => {
                [Some(mem as u32), Some(arg as u32), None]
            }
            Op::Store { reg, mem }
            | Op::Load { reg, mem }
            | Op::CopyRegMemChoice {
                out: mem, arg: reg, ..
            } => [Some(reg as u32), Some(mem as u32), None],
            Op::MinRegRegChoice { reg, arg, .. }
            | Op::MaxRegRegChoice { reg, arg, .. } => {
                [Some(reg as u32), Some(arg as u32), None]
            }
            Op::MinRegImmChoice { reg, .. }
            | Op::MaxRegImmChoice { reg, .. } => [Some(reg as u32), None, None],
        }
        .into_iter()
        .flatten()
    }

    /// Returns a `choice` field, if present
    pub fn choice(&self) -> Option<ChoiceIndex> {
        match self {
            Op::MinMemImmChoice { choice, .. }
            | Op::MaxMemImmChoice { choice, .. }
            | Op::CopyImmMemChoice { choice, .. }
            | Op::CopyImmRegChoice { choice, .. }
            | Op::MinMemRegChoice { choice, .. }
            | Op::MaxMemRegChoice { choice, .. }
            | Op::CopyRegMemChoice { choice, .. }
            | Op::CopyRegRegChoice { choice, .. }
            | Op::MinRegRegChoice { choice, .. }
            | Op::MaxRegRegChoice { choice, .. }
            | Op::MinRegImmChoice { choice, .. }
            | Op::MaxRegImmChoice { choice, .. } => Some(*choice),
            Op::Input { .. }
            | Op::Var { .. }
            | Op::CopyImm { .. }
            | Op::Load { .. }
            | Op::NegReg { .. }
            | Op::AbsReg { .. }
            | Op::RecipReg { .. }
            | Op::SqrtReg { .. }
            | Op::SquareReg { .. }
            | Op::AddRegImm { .. }
            | Op::MulRegImm { .. }
            | Op::DivRegImm { .. }
            | Op::DivImmReg { .. }
            | Op::SubImmReg { .. }
            | Op::SubRegImm { .. }
            | Op::MinRegImm { .. }
            | Op::MaxRegImm { .. }
            | Op::CopyReg { .. }
            | Op::AddRegReg { .. }
            | Op::MulRegReg { .. }
            | Op::DivRegReg { .. }
            | Op::SubRegReg { .. }
            | Op::MinRegReg { .. }
            | Op::MaxRegReg { .. }
            | Op::Store { .. } => None,
        }
    }

    /// Returns a mutable reference to the `choice` field, if present
    pub fn choice_mut(&mut self) -> Option<&mut ChoiceIndex> {
        match self {
            Op::MinMemImmChoice { choice, .. }
            | Op::MaxMemImmChoice { choice, .. }
            | Op::CopyImmMemChoice { choice, .. }
            | Op::CopyImmRegChoice { choice, .. }
            | Op::MinMemRegChoice { choice, .. }
            | Op::MaxMemRegChoice { choice, .. }
            | Op::CopyRegMemChoice { choice, .. }
            | Op::CopyRegRegChoice { choice, .. }
            | Op::MinRegRegChoice { choice, .. }
            | Op::MaxRegRegChoice { choice, .. }
            | Op::MinRegImmChoice { choice, .. }
            | Op::MaxRegImmChoice { choice, .. } => Some(choice),
            Op::Input { .. }
            | Op::Var { .. }
            | Op::CopyImm { .. }
            | Op::Load { .. }
            | Op::NegReg { .. }
            | Op::AbsReg { .. }
            | Op::RecipReg { .. }
            | Op::SqrtReg { .. }
            | Op::SquareReg { .. }
            | Op::AddRegImm { .. }
            | Op::MulRegImm { .. }
            | Op::DivRegImm { .. }
            | Op::DivImmReg { .. }
            | Op::SubImmReg { .. }
            | Op::SubRegImm { .. }
            | Op::MinRegImm { .. }
            | Op::MaxRegImm { .. }
            | Op::CopyReg { .. }
            | Op::AddRegReg { .. }
            | Op::MulRegReg { .. }
            | Op::DivRegReg { .. }
            | Op::SubRegReg { .. }
            | Op::MinRegReg { .. }
            | Op::MaxRegReg { .. }
            | Op::Store { .. } => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_vm_op_size() {
        assert_eq!(std::mem::size_of::<Op>(), 16);
    }
}
