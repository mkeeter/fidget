//! Instruction tapes in the form of assembly for a simple virtual machine
mod alloc;
mod builder;
mod eval;
mod lru;
mod op;

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
