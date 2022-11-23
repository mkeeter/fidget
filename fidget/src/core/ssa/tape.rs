use crate::{
    ssa::Op,
    vm::{RegisterAllocator, Tape as VmTape},
};

use std::{collections::BTreeMap, sync::Arc};

/// Instruction tape, storing [`Op`](crate::ssa::Op) in SSA form
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
pub struct Tape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<Op>,

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

    /// Mapping from variable names (in the original
    /// [`Context`](crate::context::Context)) to indexes in the variable array
    /// used during evaluation.
    ///
    /// This is an `Arc` so it can be trivially shared by all of the tape's
    /// descendents, since the variable array order does not change.
    pub vars: Arc<BTreeMap<String, u32>>,
}

impl Tape {
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
                Op::Input => {
                    let i = next();
                    let out = next();
                    println!("${out} = INPUT {i}");
                }
                Op::Var => {
                    let i = next();
                    let out = next();
                    println!("${out} = VAR {i}");
                }
                Op::NegReg
                | Op::AbsReg
                | Op::RecipReg
                | Op::SqrtReg
                | Op::CopyReg
                | Op::SquareReg => {
                    let arg = next();
                    let out = next();
                    let op = match op {
                        Op::NegReg => "NEG",
                        Op::AbsReg => "ABS",
                        Op::RecipReg => "RECIP",
                        Op::SqrtReg => "SQRT",
                        Op::SquareReg => "SQUARE",
                        Op::CopyReg => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${arg}");
                }

                Op::AddRegReg
                | Op::MulRegReg
                | Op::DivRegReg
                | Op::SubRegReg
                | Op::MinRegReg
                | Op::MaxRegReg => {
                    let rhs = next();
                    let lhs = next();
                    let out = next();
                    let op = match op {
                        Op::AddRegReg => "ADD",
                        Op::MulRegReg => "MUL",
                        Op::DivRegReg => "DIV",
                        Op::SubRegReg => "SUB",
                        Op::MinRegReg => "MIN",
                        Op::MaxRegReg => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                Op::AddRegImm
                | Op::MulRegImm
                | Op::DivRegImm
                | Op::DivImmReg
                | Op::SubImmReg
                | Op::SubRegImm
                | Op::MinRegImm
                | Op::MaxRegImm => {
                    let imm = f32::from_bits(next());
                    let arg = next();
                    let out = next();
                    let (op, swap) = match op {
                        Op::AddRegImm => ("ADD", false),
                        Op::MulRegImm => ("MUL", false),
                        Op::DivImmReg => ("DIV", true),
                        Op::DivRegImm => ("DIV", false),
                        Op::SubImmReg => ("SUB", true),
                        Op::SubRegImm => ("SUB", false),
                        Op::MinRegImm => ("MIN", false),
                        Op::MaxRegImm => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                Op::CopyImm => {
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
    /// to use [`simplify`](crate::tape::Tape::simplify), which simultaneously
    /// simplifies **and** performs register allocation in a single pass.
    pub fn get_asm(&self, reg_limit: u8) -> VmTape {
        let mut alloc = RegisterAllocator::new(reg_limit, self.tape.len());
        let mut data = self.data.iter();
        for &op in self.tape.iter() {
            let index = *data.next().unwrap();

            match op {
                Op::Input => {
                    let i = *data.next().unwrap();
                    alloc.op_input(index, i.try_into().unwrap());
                }
                Op::Var => {
                    let i = *data.next().unwrap();
                    alloc.op_var(index, i);
                }
                Op::CopyImm => {
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_copy_imm(index, imm);
                }
                Op::CopyReg
                | Op::NegReg
                | Op::AbsReg
                | Op::RecipReg
                | Op::SqrtReg
                | Op::SquareReg => {
                    let arg = *data.next().unwrap();
                    alloc.op_reg(index, arg, op);
                }
                Op::MinRegImm
                | Op::MaxRegImm
                | Op::AddRegImm
                | Op::MulRegImm
                | Op::DivRegImm
                | Op::DivImmReg
                | Op::SubRegImm
                | Op::SubImmReg => {
                    let arg = *data.next().unwrap();
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_reg_imm(index, arg, imm, op);
                }
                Op::AddRegReg
                | Op::MulRegReg
                | Op::DivRegReg
                | Op::SubRegReg
                | Op::MinRegReg
                | Op::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    alloc.op_reg_reg(index, lhs, rhs, op);
                }
            }
        }
        alloc.finalize()
    }
}
