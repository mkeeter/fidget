use crate::{
    tape::TapeOp,
    vm::{AsmTape, RegisterAllocator},
};

/// Instruction tape, storing [`TapeOp`](crate::tape::TapeOp) in SSA form
///
/// Each operation has the following parameters
/// - 4-byte opcode (required)
/// - 4-byte output register (required)
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs, arguments, and immediates are packed into the `data` array
///
/// All register addressing is absolute.
#[derive(Clone, Debug, Default)]
pub struct SsaTape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<TapeOp>,

    /// Variable-length data for tape clauses.
    ///
    /// Data is densely packed in the order
    /// - output slot
    /// - lhs slot (or input)
    /// - rhs slot (or immediate)
    ///
    /// i.e. a unary operation would only store two items in this array
    pub data: Vec<u32>,

    /// Number of choice operations in the tape
    pub choice_count: usize,
}

impl SsaTape {
    /// Returns the number of opcodes in the tape
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    /// Resets to an empty tape, preserving allocations
    pub fn reset(&mut self) {
        self.data.clear();
        self.tape.clear();
        self.choice_count = 0;
    }
    pub fn pretty_print(&self) {
        let mut data = self.data.iter().rev();
        let mut next = || *data.next().unwrap();
        for &op in self.tape.iter().rev() {
            match op {
                TapeOp::Input => {
                    let i = next();
                    let out = next();
                    println!("${out} = %{i}");
                }
                TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::CopyReg
                | TapeOp::SquareReg => {
                    let arg = next();
                    let out = next();
                    let op = match op {
                        TapeOp::NegReg => "NEG",
                        TapeOp::AbsReg => "ABS",
                        TapeOp::RecipReg => "RECIP",
                        TapeOp::SqrtReg => "SQRT",
                        TapeOp::SquareReg => "SQUARE",
                        TapeOp::CopyReg => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${arg}");
                }

                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::DivRegReg
                | TapeOp::SubRegReg
                | TapeOp::MinRegReg
                | TapeOp::MaxRegReg => {
                    let rhs = next();
                    let lhs = next();
                    let out = next();
                    let op = match op {
                        TapeOp::AddRegReg => "ADD",
                        TapeOp::MulRegReg => "MUL",
                        TapeOp::DivRegReg => "DIV",
                        TapeOp::SubRegReg => "SUB",
                        TapeOp::MinRegReg => "MIN",
                        TapeOp::MaxRegReg => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg
                | TapeOp::SubImmReg
                | TapeOp::SubRegImm
                | TapeOp::MinRegImm
                | TapeOp::MaxRegImm => {
                    let imm = f32::from_bits(next());
                    let arg = next();
                    let out = next();
                    let (op, swap) = match op {
                        TapeOp::AddRegImm => ("ADD", false),
                        TapeOp::MulRegImm => ("MUL", false),
                        TapeOp::DivImmReg => ("DIV", true),
                        TapeOp::DivRegImm => ("DIV", false),
                        TapeOp::SubImmReg => ("SUB", true),
                        TapeOp::SubRegImm => ("SUB", false),
                        TapeOp::MinRegImm => ("MIN", false),
                        TapeOp::MaxRegImm => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                TapeOp::CopyImm => {
                    let imm = f32::from_bits(next());
                    let out = next();
                    println!("${out} = COPY {imm}");
                }
            }
        }
    }

    /// Lowers the tape to assembly with a particular register limit
    ///
    /// Note that if you _also_ want to simplify the tape, it's more efficient
    /// to use [`simplify`](Self::simplify), which simultaneously simplifies
    /// **and** performs register allocation in a single pass.
    pub fn get_asm(&self, reg_limit: u8) -> AsmTape {
        let mut alloc = RegisterAllocator::new(reg_limit, self.tape.len());
        let mut data = self.data.iter();
        for &op in self.tape.iter() {
            let index = *data.next().unwrap();

            match op {
                TapeOp::Input => {
                    let i = *data.next().unwrap();
                    alloc.op_input(index, i.try_into().unwrap());
                }
                TapeOp::CopyImm => {
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_copy_imm(index, imm);
                }
                TapeOp::CopyReg
                | TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::SquareReg => {
                    let arg = *data.next().unwrap();
                    alloc.op_reg(index, arg, op);
                }
                TapeOp::MinRegImm
                | TapeOp::MaxRegImm
                | TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg
                | TapeOp::SubRegImm
                | TapeOp::SubImmReg => {
                    let arg = *data.next().unwrap();
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_reg_imm(index, arg, imm, op);
                }
                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::DivRegReg
                | TapeOp::SubRegReg
                | TapeOp::MinRegReg
                | TapeOp::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    alloc.op_reg_reg(index, lhs, rhs, op);
                }
            }
        }
        alloc.finalize()
    }
}
