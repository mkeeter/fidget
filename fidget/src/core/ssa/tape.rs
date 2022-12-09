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
        self.tape.clear();
        self.choice_count = 0;
    }
    pub fn pretty_print(&self) {
        for &op in self.tape.iter().rev() {
            match op {
                Op::Input(out, i) => {
                    println!("${out} = INPUT {i}");
                }
                Op::Var(out, i) => {
                    println!("${out} = VAR {i}");
                }
                Op::NegReg(out, arg)
                | Op::AbsReg(out, arg)
                | Op::RecipReg(out, arg)
                | Op::SqrtReg(out, arg)
                | Op::CopyReg(out, arg)
                | Op::SquareReg(out, arg) => {
                    let op = match op {
                        Op::NegReg(..) => "NEG",
                        Op::AbsReg(..) => "ABS",
                        Op::RecipReg(..) => "RECIP",
                        Op::SqrtReg(..) => "SQRT",
                        Op::SquareReg(..) => "SQUARE",
                        Op::CopyReg(..) => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${arg}");
                }

                Op::AddRegReg(out, lhs, rhs)
                | Op::MulRegReg(out, lhs, rhs)
                | Op::DivRegReg(out, lhs, rhs)
                | Op::SubRegReg(out, lhs, rhs)
                | Op::MinRegReg(out, lhs, rhs)
                | Op::MaxRegReg(out, lhs, rhs) => {
                    let op = match op {
                        Op::AddRegReg(..) => "ADD",
                        Op::MulRegReg(..) => "MUL",
                        Op::DivRegReg(..) => "DIV",
                        Op::SubRegReg(..) => "SUB",
                        Op::MinRegReg(..) => "MIN",
                        Op::MaxRegReg(..) => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                Op::AddRegImm(out, arg, imm)
                | Op::MulRegImm(out, arg, imm)
                | Op::DivRegImm(out, arg, imm)
                | Op::DivImmReg(out, arg, imm)
                | Op::SubImmReg(out, arg, imm)
                | Op::SubRegImm(out, arg, imm)
                | Op::MinRegImm(out, arg, imm)
                | Op::MaxRegImm(out, arg, imm) => {
                    let (op, swap) = match op {
                        Op::AddRegImm(..) => ("ADD", false),
                        Op::MulRegImm(..) => ("MUL", false),
                        Op::DivImmReg(..) => ("DIV", true),
                        Op::DivRegImm(..) => ("DIV", false),
                        Op::SubImmReg(..) => ("SUB", true),
                        Op::SubRegImm(..) => ("SUB", false),
                        Op::MinRegImm(..) => ("MIN", false),
                        Op::MaxRegImm(..) => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                Op::CopyImm(out, imm) => {
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
        for &op in self.tape.iter() {
            alloc.op(op)
        }
        alloc.finalize()
    }
}
