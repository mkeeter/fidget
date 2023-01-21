use crate::context::{BinaryOpcode, UnaryOpcode};

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
    Input(T, u8),
    Var(T, u32),
    CopyImm(T, f32),
    Load(T, u32),
    Store(T, u32),
    Reg(UnaryOpcode, T, T),
    RegImm(BinaryOpcode, T, T, f32),
    ImmReg(BinaryOpcode, T, T, f32),
    RegReg(BinaryOpcode, T, T, T),
}

impl<T: Copy> Op<T> {
    pub fn output(&self) -> Option<T> {
        match self {
            Op::Input(out, ..)
            | Op::Var(out, ..)
            | Op::CopyImm(out, ..)
            | Op::Reg(_, out, ..)
            | Op::RegImm(_, out, ..)
            | Op::ImmReg(_, out, ..)
            | Op::RegReg(_, out, ..) => Some(*out),
            Op::Load(..) | Op::Store(..) => None,
        }
    }
    pub fn choice_count(&self) -> usize {
        match self {
            Op::RegImm(BinaryOpcode::Min | BinaryOpcode::Max, ..)
            | Op::ImmReg(BinaryOpcode::Min | BinaryOpcode::Max, ..)
            | Op::RegReg(BinaryOpcode::Min | BinaryOpcode::Max, ..) => 1,

            _ => 0,
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
