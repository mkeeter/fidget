//! Instruction tapes in the form of assembly for a simple virtual machine
mod alloc;
mod builder;
mod eval;
mod lru;
mod op;

use crate::context::{BinaryOpcode, UnaryOpcode};
pub(super) use alloc::RegisterAllocator;
pub(super) use builder::Builder;

pub use eval::Eval;
pub use op::Op;

/// Low-level tape for use with the Fidget virtual machine (or to be lowered
/// further into machine instructions).
#[derive(Clone, Default)]
pub struct Tape {
    pub(crate) tape: Vec<Op>,

    /// Total allocated slots
    pub(crate) slot_count: u32,

    /// Total number of choices
    pub(crate) choice_count: usize,

    /// Number of registers, before we fall back to Load/Store operations
    reg_limit: u8,
}

impl Tape {
    pub fn new(reg_limit: u8) -> Self {
        Self {
            tape: vec![],
            slot_count: 1,
            choice_count: 0,
            reg_limit,
        }
    }
    /// Resets this tape, retaining its allocations
    pub fn reset(&mut self, reg_limit: u8) {
        self.tape.clear();
        self.slot_count = 1;
        self.reg_limit = reg_limit;
    }
    /// Returns the register limit with which this tape was planned
    pub fn reg_limit(&self) -> u8 {
        self.reg_limit
    }
    /// Returns the number of unique register and memory locations that are used
    /// by this tape.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count as usize
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Op> {
        self.tape.iter()
    }
    #[inline]
    pub(crate) fn push(&mut self, op: Op) {
        self.tape.push(op)
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
                Op::CopyImm(out, f) => {
                    println!("${out} = {f}");
                }
                Op::Reg(op, out, arg) => {
                    let op = match op {
                        UnaryOpcode::Neg => "NEG",
                        UnaryOpcode::Abs => "ABS",
                        UnaryOpcode::Recip => "RECIP",
                        UnaryOpcode::Sqrt => "SQRT",
                        UnaryOpcode::Square => "SQUARE",
                        UnaryOpcode::Copy => "COPY",
                    };
                    println!("${out} = {op} ${arg}");
                }

                Op::RegReg(op, out, lhs, rhs) => {
                    let op = match op {
                        BinaryOpcode::Add => "ADD",
                        BinaryOpcode::Mul => "MUL",
                        BinaryOpcode::Div => "DIV",
                        BinaryOpcode::Sub => "SUB",
                        BinaryOpcode::Min => "MIN",
                        BinaryOpcode::Max => "MAX",
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                Op::RegImm(op, out, arg, imm) => {
                    // TODO: duplicate code
                    let op = match op {
                        BinaryOpcode::Add => "ADD",
                        BinaryOpcode::Mul => "MUL",
                        BinaryOpcode::Div => "DIV",
                        BinaryOpcode::Sub => "SUB",
                        BinaryOpcode::Min => "MIN",
                        BinaryOpcode::Max => "MAX",
                    };
                    println!("${out} = {op} ${arg} {imm}");
                }
                Op::ImmReg(op, out, arg, imm) => {
                    let op = match op {
                        BinaryOpcode::Add => "ADD",
                        BinaryOpcode::Mul => "MUL",
                        BinaryOpcode::Div => "DIV",
                        BinaryOpcode::Sub => "SUB",
                        BinaryOpcode::Min => "MIN",
                        BinaryOpcode::Max => "MAX",
                    };
                    println!("${out} = {op} {imm} ${arg}");
                }
                Op::Load(reg, mem) => {
                    println!("${reg} <= %{mem}");
                }
                Op::Store(reg, mem) => {
                    println!("%{mem} <= ${reg}");
                }
            }
        }
    }
}
