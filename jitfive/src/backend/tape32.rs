use std::collections::{BTreeMap, BTreeSet};

use crate::{
    backend::common::{NodeIndex, Op},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    scheduled::Scheduled,
    util::{bimap::Bimap, queue::PriorityQueue},
};

use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Copy, Clone, Debug, ToPrimitive, FromPrimitive)]
pub enum ClauseOp32 {
    /// Load from an extended register to a main register
    Load,

    /// Store from a main register to an extended register
    Store,

    /// Swap from a main register to an extended register
    Swap,

    /// Reads one of the inputs (i.e. X or Y)
    Input,

    CopyReg,
    NegReg,
    AbsReg,
    RecipReg,
    SqrtReg,
    SquareReg,

    AddRegReg,
    MulRegReg,
    SubRegReg,
    MinRegReg,
    MaxRegReg,
}

impl ClauseOp32 {
    fn name(&self) -> &'static str {
        match self {
            ClauseOp32::Load => "LOAD",
            ClauseOp32::Store => "STORE",
            ClauseOp32::Swap => "SWAP",
            ClauseOp32::Input => "INPUT",
            ClauseOp32::CopyReg => "COPY",
            ClauseOp32::NegReg => "NEG",
            ClauseOp32::AbsReg => "ABS",
            ClauseOp32::RecipReg => "RECIP",
            ClauseOp32::SqrtReg => "SQRT",
            ClauseOp32::SquareReg => "SQUARE",
            ClauseOp32::AddRegReg => "ADD",
            ClauseOp32::MulRegReg => "MUL",
            ClauseOp32::SubRegReg => "SUB",
            ClauseOp32::MinRegReg => "MIN",
            ClauseOp32::MaxRegReg => "MAX",
        }
    }
    fn as_reg_imm(&self) -> ClauseOp64 {
        match self {
            ClauseOp32::AddRegReg => ClauseOp64::AddRegImm,
            ClauseOp32::SubRegReg => ClauseOp64::SubRegImm,
            ClauseOp32::MulRegReg => ClauseOp64::MulRegImm,
            ClauseOp32::MinRegReg => ClauseOp64::MinRegImm,
            ClauseOp32::MaxRegReg => ClauseOp64::MaxRegImm,
            _ => panic!(),
        }
    }
    fn as_imm_reg(&self) -> ClauseOp64 {
        match self {
            ClauseOp32::AddRegReg => ClauseOp64::AddRegImm,
            ClauseOp32::SubRegReg => ClauseOp64::SubImmReg,
            ClauseOp32::MulRegReg => ClauseOp64::MulRegImm,
            ClauseOp32::MinRegReg => ClauseOp64::MinRegImm,
            ClauseOp32::MaxRegReg => ClauseOp64::MaxRegImm,
            _ => panic!(),
        }
    }
}

impl From<BinaryOpcode> for ClauseOp32 {
    fn from(b: BinaryOpcode) -> Self {
        match b {
            BinaryOpcode::Sub => ClauseOp32::SubRegReg,
            BinaryOpcode::Add => ClauseOp32::AddRegReg,
            BinaryOpcode::Mul => ClauseOp32::MulRegReg,
        }
    }
}

impl From<BinaryChoiceOpcode> for ClauseOp32 {
    fn from(b: BinaryChoiceOpcode) -> Self {
        match b {
            BinaryChoiceOpcode::Min => ClauseOp32::MinRegReg,
            BinaryChoiceOpcode::Max => ClauseOp32::MaxRegReg,
        }
    }
}

impl From<UnaryOpcode> for ClauseOp32 {
    fn from(b: UnaryOpcode) -> Self {
        match b {
            UnaryOpcode::Square => ClauseOp32::SquareReg,
            UnaryOpcode::Sqrt => ClauseOp32::SqrtReg,
            UnaryOpcode::Abs => ClauseOp32::AbsReg,
            UnaryOpcode::Recip => ClauseOp32::RecipReg,
            UnaryOpcode::Neg => ClauseOp32::NegReg,
        }
    }
}

#[derive(Copy, Clone, Debug, ToPrimitive, FromPrimitive)]
pub enum ClauseOp64 {
    /// Add a register and an immediate
    AddRegImm,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Subtract a register from an immediate
    SubImmReg,
    /// Subtract an immediate from a register
    SubRegImm,
    /// Compute the minimum of a register and an immediate
    MinRegImm,
    /// Compute the maximum of a register and an immediate
    MaxRegImm,

    /// Copy an immediate to a register
    CopyImm,
}

impl ClauseOp64 {
    fn name(&self) -> &'static str {
        match self {
            ClauseOp64::CopyImm => "COPY",
            ClauseOp64::AddRegImm => "ADD",
            ClauseOp64::MulRegImm => "MUL",
            ClauseOp64::SubImmReg | ClauseOp64::SubRegImm => "SUB",
            ClauseOp64::MinRegImm => "MIN",
            ClauseOp64::MaxRegImm => "MAX",
        }
    }
}

const NEXT: u32 = 1 << 31;
const LONG: u32 = 1 << 30;

/// The instruction tape is given as a `Vec<u32>`, which can be thought of as
/// bytecode or instructions for a very simple virtual machine.
///
/// Every `u32` in the tape is either an instruction or an `f32` immediate
/// (passed as raw bits).
///
/// To disambiguate, we use two bits as flags:
///
/// | Bits | Name   | Meaning                        |
/// |------|--------|--------------------------------|
/// | 31   | `NEXT` | Next instruction is 64-bit     |
/// | 30   | `LONG` | This instruction is 64-bit     |
///
/// We also guarantee that the item at index 0 is a 32-bit operation, which
/// allows us to unambiguously read the tape in either direction.
///
/// Here is an example tape encoding the operation
/// ```
/// # let x = 1.0;
/// # let y = 1.0;
/// # let max = |x: f32, y: f32| x.max(y);
/// # let square = |x: f32| x * x;
/// max(square(x) + square(y) - 0.5, 0.25 - (square(x) + square(y)))
/// # ;
/// ```
///
/// The tape is evaluated from back to front, i.e. starting at index 10 and
/// moving up through index 0.
///
/// | Index | Operation |  Out   | LHS   | RHS   | Value | `LONG` | `NEXT` |
/// |-------|-----------|--------|-------|-------|-------|--------|--------|
/// | 0     | `COPY`    | `$0`   | `$1`  | -     | -     |        |        |
/// | 1     | `MAX`     | `$1`   | `$0`  | `$1`  | -     |        | ✓      |
/// | 2     | -         | -      | -     | -     | 0.25  |        |        |
/// | 3     | `SUB`     | `$1`   | -     | `$1`  | -     | ✓      | ✓      |
/// | 4     | -         | -      | -     | -     | 0.5   |        |        |
/// | 5     | `SUB`     | `$0`   | `$1`  | -     | -     | ✓      |        |
/// | 6     | `ADD`     | `$1`   | `$0`  | `$1`  |       |        |        |
/// | 7     | `SQUARE`  | `$1`   | `$1`  | -     | -     |        |        |
/// | 8     | `INPUT`   | `$1`   | `%1`  | -     | -     |        |        |
/// | 9     | `SQUARE`  | `$0`   | `$0`  | -     | -     |        |        |
/// | 10    | `INPUT`   | `$0`   | `%0`  | -     | -     |        |        |
///
/// Registers are shown with `$`; input arguments use `%`.
///
/// # 32-bit instructions
/// ## Common operations on registers
/// For a 32-bit instruction (i.e. bit 30 is 0), there are three encodings.  The
/// most common is a operation on **registers**:
///
/// | 31     | 30         | 29-24  | 23-16 | 18-8 | 7-0  |
/// |--------|------------|--------|-------|------|------|
/// | `NEXT` | `LONG` (0) | Opcode | LHS   | RHS  | Out  |
///
/// This allows us to encode the 64 most common opcodes into a single `u32`,
/// using up to 256 registers.
///
/// ## Input
/// The encoding for `Input` operations uses bits 23-16 as an index into the
/// function's argument list.  Right now, it must be 0 (for X) or 1 (for Y).
///
/// | 31     | 30         | 29-24            | 23-16 | 18-8 | 7-0  |
/// |--------|------------|------------------|-------|------|------|
/// | `NEXT` | `LONG` (0) | Opcode (`Input`) | Index | -    | Out  |
///
/// ## Load and store
/// The common operation encoding limits us to 256 registers; if we need more
/// storage space, then we can use `Load`, `Store`, and `Swap` to shuffle data
/// around into 16384 extra locations (referred to as "memory").
///
/// Registers and memory together are referred to as "slots", e.g. in the
/// `total_slots` member variable.
///
/// The lowest 256 memory locations are unused, so that we can use a single
/// array for both registers _and_ memory when appropriate.
///
/// (If the `Tape` is constructed with a lower register limit, then memory
/// begins at that value instead of at slot 256)
///
/// `Load` and `Store` are implemented as a 32-bit operation with a slightly
/// different encoding
///
/// | 31     | 30         | 29-24  | 23-8   |  7-0     |
/// |--------|------------|--------|--------|----------|
/// | `NEXT` | `LONG` (0) | Opcode | Memory | Register |
///
/// (where the opcode is limited to `Load`, `Store`, and `Swap`)
///
/// # 64-bit instructions
/// ## Immediate form
/// | 31     | 30         | 29-16  | 15-8 |  7-0 |
/// |--------|------------|--------|------|------|
/// | `NEXT` | `LONG` (1) | Opcode | Arg  | Out  |
///
/// It is preceeded by an immediate (an `f32` as raw bits), as shown in the
/// example tape above.
#[derive(Debug)]
pub struct Tape {
    pub tape: Vec<u32>,

    /// Total number of slots required for evaluation, including both registers
    /// and memory.
    pub total_slots: usize,

    /// Number of registers in use
    ///
    /// This is always `<= total_slots`
    pub reg_count: usize,

    /// The number of nodes which push to a choice array
    pub choice_count: usize,
}

/// Workspace to evaluate a tape
pub struct TapeEval<'a> {
    tape: &'a Tape,
    slots: Vec<f32>,
    choices: Vec<Choice>,
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum Choice {
    Left,
    Right,
    Both,
}

impl Tape {
    /// Build a new tape from a pre-scheduled set of instructions
    pub fn new(t: &Scheduled) -> Self {
        Self::from_builder(TapeBuilder::new(t))
    }

    /// Build a new tape from a pre-scheduled set of instructions, with a
    /// limited number of fast registers (instead of 256).
    pub fn new_with_reg_limit(t: &Scheduled, reg_limit: usize) -> Self {
        Self::from_builder(TapeBuilder::new_with_reg_limit(t, reg_limit))
    }

    fn from_builder(mut builder: TapeBuilder) -> Self {
        builder.run();
        builder.out.reverse();
        Self {
            tape: builder.out,
            reg_count: builder.reg_limit.min(builder.total_slots),
            total_slots: builder.total_slots as usize,
            choice_count: builder.choice_count,
        }
    }

    /// Builds an evaluator which takes a (read-only) reference to this tape
    pub fn to_evaluator(&self) -> TapeEval {
        // Bias towards 256-byte chunks of vector, to (maybe) ease pressure on
        // the allocator and persuade it to reuse them!
        let mut slots = Vec::with_capacity(self.total_slots.max(64));
        slots.resize(self.total_slots, 0.0);
        let choices = Vec::with_capacity(self.choice_count.max(256));
        TapeEval {
            tape: self,
            slots,
            choices,
        }
    }

    // TODO: test MAX(X + 1, Y) => X + 1
    // (since that requires injecting a copy)
    pub fn simplify(&self, mut choices: &[Choice]) -> Self {
        let mut out = vec![];
        let mut next_was_set = false;

        let mut active = vec![false; self.total_slots];
        active[0] = true;

        let mut choice_count = 0;

        let mut iter = self.tape.iter().cloned();
        while let Some(next) = iter.next() {
            let (mut v, mut imm) = if next_was_set {
                (iter.next().unwrap(), Some(next))
            } else {
                (next, None)
            };
            next_was_set = (v & NEXT) != 0;

            // In general, the output register is in the bottom 8 bits of the
            // value.  There are a few exceptions for load/store/swap, which are
            // special cases!
            let is_64bit = (v & LONG) != 0;
            let is_active = if is_64bit {
                let out_reg = v & 0xFF;
                active[out_reg as usize]
            } else {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp32::from_u32(op).unwrap();
                match op {
                    ClauseOp32::Store => {
                        let out_mem = (v >> 8) & 0xFFFF;
                        active[out_mem as usize]
                    }
                    ClauseOp32::Swap => {
                        let out_mem = (v >> 8) & 0xFFFF;
                        let out_reg = v & 0xFF;
                        active[out_mem as usize] | active[out_reg as usize]
                    }
                    _ => {
                        // Every other `ClauseOp32` opcode encodes the target
                        // registers in the lowest byte.
                        let out_reg = v & 0xFF;
                        active[out_reg as usize]
                    }
                }
            };

            if !is_active {
                continue;
            }

            // 64-bit opcode
            if is_64bit {
                assert!(imm.is_some());
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp64::from_u32(op).unwrap();
                let out_reg = (v & 0xFF) as usize;
                match op {
                    ClauseOp64::AddRegImm
                    | ClauseOp64::MulRegImm
                    | ClauseOp64::SubImmReg
                    | ClauseOp64::SubRegImm => {
                        let arg_reg = ((v >> 8) & 0xFF) as usize;
                        active[out_reg] = false;
                        active[arg_reg] = true;
                    }
                    ClauseOp64::CopyImm => {
                        active[out_reg] = false;
                    }
                    ClauseOp64::MinRegImm | ClauseOp64::MaxRegImm => {
                        let arg_reg = ((v >> 8) & 0xFF) as usize;
                        active[out_reg] = false;
                        let (c, choices_) = choices.split_last().unwrap();
                        choices = choices_;
                        match c {
                            // Left always refers to the register argument
                            Choice::Left => {
                                active[arg_reg] = true;
                                // This becomes a CopyReg 32-bit operation
                                if out_reg == arg_reg {
                                    // Skip entirely, rather than a dummy copy
                                    continue;
                                } else {
                                    // Switch over to a 32-bit CopyReg op,
                                    // and drop the immediate (since it's now
                                    // unused).
                                    v = ((ClauseOp32::CopyReg as u32) << 24)
                                        | ((arg_reg as u32) << 16)
                                        | (out_reg as u32);
                                    imm = None;
                                }
                            }
                            // Left always refers to the immediate argument
                            Choice::Right => {
                                // This becomes a CopyImm 32-bit operation
                                v = LONG
                                    | ((ClauseOp64::CopyImm as u32) << 16)
                                    | (out_reg as u32);
                            }
                            Choice::Both => {
                                choice_count += 1;
                                active[arg_reg] = true;
                                // Leave v unchanged
                            }
                        }
                    }
                }
            } else {
                assert!(imm.is_none());
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp32::from_u32(op).unwrap();
                match op {
                    ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                        let fast_reg = (v & 0xFF) as usize;
                        let extended_reg = ((v >> 8) & 0xFFFF) as usize;

                        match op {
                            ClauseOp32::Load => {
                                active[fast_reg] = false;
                                active[extended_reg] = true;
                            }
                            ClauseOp32::Store => {
                                active[extended_reg] = false;
                                active[fast_reg] = true;
                            }
                            ClauseOp32::Swap => {
                                active[fast_reg] = true;
                                active[extended_reg] = true;
                            }
                            _ => unreachable!(),
                        }
                    }
                    ClauseOp32::Input => {
                        let out_reg = (v & 0xFF) as usize;
                        active[out_reg] = false;
                    }
                    ClauseOp32::CopyReg
                    | ClauseOp32::NegReg
                    | ClauseOp32::AbsReg
                    | ClauseOp32::RecipReg
                    | ClauseOp32::SqrtReg
                    | ClauseOp32::SquareReg => {
                        let lhs_reg = ((v >> 16) & 0xFF) as usize;
                        let out_reg = (v & 0xFF) as usize;
                        active[out_reg] = false;
                        active[lhs_reg] = true;
                    }
                    ClauseOp32::AddRegReg
                    | ClauseOp32::MulRegReg
                    | ClauseOp32::SubRegReg => {
                        let lhs_reg = ((v >> 16) & 0xFF) as usize;
                        let rhs_reg = ((v >> 8) & 0xFF) as usize;
                        let out_reg = (v & 0xFF) as usize;
                        active[out_reg] = false;
                        active[lhs_reg] = true;
                        active[rhs_reg] = true;
                    }
                    ClauseOp32::MinRegReg | ClauseOp32::MaxRegReg => {
                        let lhs_reg = ((v >> 16) & 0xFF) as usize;
                        let rhs_reg = ((v >> 8) & 0xFF) as usize;
                        let out_reg = (v & 0xFF) as usize;
                        active[out_reg] = false;
                        let (c, choices_) = choices.split_last().unwrap();
                        choices = choices_;
                        match c {
                            Choice::Left => {
                                active[lhs_reg] = true;
                                if lhs_reg == out_reg {
                                    // Skip entirely, rather than a dummy copy
                                    continue;
                                } else {
                                    v = ((ClauseOp32::CopyReg as u32) << 24)
                                        | ((lhs_reg as u32) << 16)
                                        | (out_reg as u32);
                                }
                            }
                            Choice::Right => {
                                active[rhs_reg] = true;
                                if rhs_reg == out_reg {
                                    // Skip entirely, rather than a dummy copy
                                    continue;
                                } else {
                                    v = ((ClauseOp32::CopyReg as u32) << 24)
                                        | ((rhs_reg as u32) << 16)
                                        | (out_reg as u32);
                                }
                            }
                            Choice::Both => {
                                active[lhs_reg] = true;
                                active[rhs_reg] = true;
                                choice_count += 1;
                                // v remains unchanged
                            }
                        }
                    }
                }
            }
            if let Some(imm) = imm {
                // If the first operation in the new tape is an immediate, then
                // push a dummy CopyReg 0 0 0 to make decoding unambiguous
                if out.is_empty() {
                    out.push((ClauseOp32::CopyReg as u32) << 24);
                }
                *out.last_mut().unwrap() |= NEXT;
                out.push(imm);
            }
            out.push(v);
        }

        Self {
            tape: out,
            total_slots: self.total_slots,
            reg_count: self.reg_count,
            choice_count,
        }
    }

    pub fn pretty_print_tape(&self) {
        let mut iter = self.tape.iter().rev();
        while let Some(v) = iter.next() {
            // 32-bit instruction
            if v & LONG == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp32::from_u32(op).unwrap();
                match op {
                    ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                        let short = v & 0xFF;
                        let ext = (v >> 8) & 0xFFFF;
                        match op {
                            ClauseOp32::Load => {
                                println!("${} = LOAD ${}", short, ext)
                            }
                            ClauseOp32::Store => {
                                println!("${} = STORE ${}", ext, short)
                            }
                            ClauseOp32::Swap => {
                                println!("SWAP ${} ${}", short, ext)
                            }
                            _ => unreachable!(),
                        }
                    }
                    ClauseOp32::Input => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let out_reg = v & 0xFF;
                        println!("${} = INPUT %{}", out_reg, lhs_reg);
                    }
                    ClauseOp32::CopyReg
                    | ClauseOp32::NegReg
                    | ClauseOp32::AbsReg
                    | ClauseOp32::RecipReg
                    | ClauseOp32::SqrtReg
                    | ClauseOp32::SquareReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let out_reg = v & 0xFF;
                        println!("${} = {} ${}", out_reg, op.name(), lhs_reg);
                    }
                    ClauseOp32::MaxRegReg
                    | ClauseOp32::MinRegReg
                    | ClauseOp32::SubRegReg
                    | ClauseOp32::AddRegReg
                    | ClauseOp32::MulRegReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let rhs_reg = (v >> 8) & 0xFF;
                        let out_reg = v & 0xFF;
                        println!(
                            "${} = {} ${} ${}",
                            out_reg,
                            op.name(),
                            lhs_reg,
                            rhs_reg
                        );
                    }
                }
            } else {
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp64::from_u32(op).unwrap();
                let next = iter.next().unwrap();
                let imm = f32::from_bits(*next);
                let arg_reg = (v >> 8) & 0xFF;
                let out_reg = v & 0xFF;
                match op {
                    ClauseOp64::MinRegImm
                    | ClauseOp64::MaxRegImm
                    | ClauseOp64::AddRegImm
                    | ClauseOp64::MulRegImm
                    | ClauseOp64::SubRegImm => {
                        println!(
                            "${} = {} ${} {}",
                            out_reg,
                            op.name(),
                            arg_reg,
                            imm
                        );
                    }
                    ClauseOp64::SubImmReg => {
                        println!(
                            "${} = {} {} ${}",
                            out_reg,
                            op.name(),
                            imm,
                            arg_reg,
                        );
                    }
                    ClauseOp64::CopyImm => {
                        println!("${} = COPY {}", out_reg, imm);
                    }
                }
            }
        }
        println!(
            "{} slots, {} instructions",
            self.total_slots,
            self.tape.len()
        );
    }
}

impl<'a> TapeEval<'a> {
    pub fn f(&mut self, x: f32, y: f32) -> f32 {
        let mut iter = self.tape.tape.iter().rev();
        while let Some(v) = iter.next() {
            // 32-bit instruction
            if v & LONG == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp32::from_u32(op).unwrap();
                match op {
                    ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                        let reg = (v & 0xFF) as usize;
                        let mem = ((v >> 8) & 0xFFFF) as usize;
                        match op {
                            ClauseOp32::Load => {
                                self.slots[reg] = self.slots[mem]
                            }
                            ClauseOp32::Store => {
                                self.slots[mem] = self.slots[reg]
                            }
                            ClauseOp32::Swap => {
                                self.slots.swap(reg, mem);
                            }
                            _ => unreachable!(),
                        }
                    }
                    ClauseOp32::Input => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let out_reg = v & 0xFF;
                        let out = match lhs_reg {
                            0 => x,
                            1 => y,
                            _ => panic!(),
                        };
                        self.slots[out_reg as usize] = out;
                    }
                    ClauseOp32::CopyReg
                    | ClauseOp32::NegReg
                    | ClauseOp32::AbsReg
                    | ClauseOp32::RecipReg
                    | ClauseOp32::SqrtReg
                    | ClauseOp32::SquareReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let lhs = self.slots[lhs_reg as usize];
                        let out_reg = v & 0xFF;
                        self.slots[out_reg as usize] = match op {
                            ClauseOp32::CopyReg => lhs,
                            ClauseOp32::NegReg => -lhs,
                            ClauseOp32::AbsReg => lhs.abs(),
                            ClauseOp32::RecipReg => 1.0 / lhs,
                            ClauseOp32::SqrtReg => lhs.sqrt(),
                            ClauseOp32::SquareReg => lhs * lhs,
                            _ => unreachable!(),
                        };
                    }

                    ClauseOp32::AddRegReg
                    | ClauseOp32::MulRegReg
                    | ClauseOp32::SubRegReg
                    | ClauseOp32::MinRegReg
                    | ClauseOp32::MaxRegReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let rhs_reg = (v >> 8) & 0xFF;
                        let out_reg = v & 0xFF;
                        let lhs = self.slots[lhs_reg as usize];
                        let rhs = self.slots[rhs_reg as usize];
                        let out = match op {
                            ClauseOp32::AddRegReg => lhs + rhs,
                            ClauseOp32::MulRegReg => lhs * rhs,
                            ClauseOp32::SubRegReg => lhs - rhs,
                            ClauseOp32::MinRegReg => lhs.min(rhs),
                            ClauseOp32::MaxRegReg => lhs.max(rhs),
                            ClauseOp32::Load | ClauseOp32::Store => {
                                unreachable!()
                            }
                            _ => unreachable!(),
                        };
                        self.slots[out_reg as usize] = out;
                    }
                }
            } else {
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp64::from_u32(op).unwrap();
                let next = iter.next().unwrap();
                let arg_reg = (v >> 8) & 0xFF;
                let out_reg = v & 0xFF;
                let arg = self.slots[arg_reg as usize];
                let imm = f32::from_bits(*next);
                let out = match op {
                    ClauseOp64::AddRegImm => arg + imm,
                    ClauseOp64::MulRegImm => arg * imm,
                    ClauseOp64::SubImmReg => imm - arg,
                    ClauseOp64::SubRegImm => arg - imm,
                    ClauseOp64::MinRegImm => arg.min(imm),
                    ClauseOp64::MaxRegImm => arg.max(imm),
                    ClauseOp64::CopyImm => imm,
                };
                self.slots[out_reg as usize] = out;
            }
        }
        self.slots[0]
    }

    /// Returns a simplified tape based on `self.choices`
    ///
    /// The choices array should have been calculated during the last interval
    /// evaluation.
    pub fn push(&self) -> Tape {
        self.tape.simplify(&self.choices)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Type-safe container for an allocated register
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Register(u8);

/// Type-safe container for an allocated slot in memory
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Memory(u16);

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Slot {
    Register(Register),
    Memory(Memory),
}

/// Helper `struct` to hold spare state when building an `Tape`
struct TapeBuilder<'a> {
    t: &'a Scheduled,

    /// Output tape, in progress
    out: Vec<u32>,

    /// Total number of memory slots
    ///
    /// This should always equal the sum of `spare_registers`, `spare_memory`,
    /// and `allocations`
    total_slots: usize,

    /// The total number of min and max clauses, which push a choice when
    /// evaluated as intervals
    choice_count: usize,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_registers: Vec<Register>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_memory: Vec<Memory>,

    /// Active nodes, sorted by position in the input data
    active: BTreeSet<(usize, NodeIndex)>,

    /// Priority queue of recently used short registers
    ///
    /// `pop()` will return the _oldest_ short register
    short_register_lru: PriorityQueue<usize, Register>,

    /// Mapping from active nodes to available slots
    allocations: Bimap<NodeIndex, Slot>,

    /// Constants declared in the compiled program
    constants: BTreeMap<NodeIndex, f32>,

    /// Position in the input data, used to determine when a node has died
    i: usize,

    /// Marks that the previous clause was 64-bit
    was_64: bool,

    /// The highest index of a register
    ///
    /// When used by the interpreter, this is 255 (since fast registers are
    /// packed into 8 bits); however, it can be reduced if we're compiling down
    /// to assembly.
    reg_limit: usize,
}

enum Allocation {
    Register(Register),
    Immediate(f32),
}

impl<'a> TapeBuilder<'a> {
    fn new(t: &'a Scheduled) -> Self {
        Self::new_with_reg_limit(t, u8::MAX as usize)
    }

    fn new_with_reg_limit(t: &'a Scheduled, reg_limit: usize) -> Self {
        Self {
            t,
            out: vec![],
            total_slots: 0,
            choice_count: 0,
            short_register_lru: PriorityQueue::new(),
            spare_registers: vec![],
            spare_memory: vec![],
            active: BTreeSet::new(),
            allocations: Bimap::new(),
            constants: BTreeMap::new(),
            i: 0,
            was_64: false,
            reg_limit,
        }
    }

    /// Claims an extended register for the given node
    fn get_memory(&mut self) -> Memory {
        if let Some(r) = self.spare_memory.pop() {
            r
        } else {
            let r = Memory(self.total_slots.try_into().unwrap());
            self.total_slots += 1;
            r
        }
    }

    /// Claims a short register
    ///
    /// This may require evicting a short register
    fn get_registers(&mut self) -> Register {
        if let Some(r) = self.spare_registers.pop() {
            r
        } else if self.total_slots <= self.reg_limit {
            let r = Register(self.total_slots as u8);
            self.total_slots += 1;
            r
        } else {
            // Evict a short register
            let target = self.short_register_lru.pop().unwrap();
            let node = self
                .allocations
                .erase_right(&Slot::Register(target))
                .unwrap();
            let ext = self.get_memory();
            self.build_store_ext(target, ext);
            let b = self.allocations.insert(node, Slot::Memory(ext));
            assert!(b);
            target
        }
    }

    /// Claims a short register and loads a value from `ext`
    ///
    /// This may require evicting a short register, in which case it is swapped
    /// with `ext` (since we know `ext` is about to be available).
    fn swap_short_register(&mut self, ext: Memory) -> Register {
        if let Some(r) = self.spare_registers.pop() {
            assert!(self.allocations.get_right(&Slot::Register(r)).is_none());
            self.build_load_ext(ext, r);
            self.release_reg(Slot::Memory(ext));
            r
        } else if self.total_slots <= self.reg_limit {
            let r = Register(self.total_slots as u8);
            self.total_slots += 1;
            self.build_load_ext(ext, r);
            self.release_reg(Slot::Memory(ext));
            r
        } else {
            // Evict a short register, reassigning its allocation data
            let target = self.short_register_lru.pop().unwrap();
            let node = self
                .allocations
                .erase_right(&Slot::Register(target))
                .unwrap();
            self.build_swap_ext(target, ext);
            self.allocations.erase_right(&Slot::Memory(ext)).unwrap();
            let b = self.allocations.insert(node, Slot::Memory(ext));
            assert!(b);
            target
        }
    }

    fn build_op_32(
        &mut self,
        op: ClauseOp32,
        lhs: Register,
        rhs: Register,
        out: Register,
    ) {
        let flag = if self.was_64 { NEXT } else { 0 };
        self.was_64 = false;

        if matches!(op, ClauseOp32::MinRegReg | ClauseOp32::MaxRegReg) {
            self.choice_count += 1;
        }

        let op = op.to_u32().unwrap();
        assert!(op < 64);
        self.out.push(
            flag | (op << 24)
                | ((lhs.0 as u32) << 16)
                | ((rhs.0 as u32) << 8)
                | (out.0 as u32),
        );
    }

    fn build_op_64(
        &mut self,
        op: ClauseOp64,
        arg: Register,
        imm: f32,
        out: Register,
    ) {
        let flag = if self.was_64 { NEXT } else { 0 };
        self.was_64 = true;

        if matches!(op, ClauseOp64::MinRegImm | ClauseOp64::MaxRegImm) {
            self.choice_count += 1;
        }

        let op = op.to_u32().unwrap();
        assert!(op < 16384);
        self.out.push(
            flag | LONG | (op << 16) | ((arg.0 as u32) << 8) | (out.0 as u32),
        );
        self.out.push(imm.to_bits());
    }

    fn build_op_reg_reg(
        &mut self,
        op: ClauseOp32,
        lhs: Register,
        rhs: Register,
        out: Register,
    ) {
        self.build_op_32(op, lhs, rhs, out)
    }

    fn build_op_reg(&mut self, op: ClauseOp32, lhs: Register, out: Register) {
        self.build_op_32(op, lhs, Register(0), out)
    }

    fn build_op_reg_imm(
        &mut self,
        op: ClauseOp32,
        lhs: Register,
        rhs: f32,
        out: Register,
    ) {
        let op = op.as_reg_imm();
        self.build_op_64(op, lhs, rhs, out);
    }

    fn build_op_imm_reg(
        &mut self,
        op: ClauseOp32,
        lhs: f32,
        rhs: Register,
        out: Register,
    ) {
        let op = op.as_imm_reg();
        self.build_op_64(op, rhs, lhs, out);
    }

    fn build_op(
        &mut self,
        op: ClauseOp32,
        lhs: Allocation,
        rhs: Allocation,
        out: Register,
    ) {
        match (lhs, rhs) {
            (Allocation::Register(lhs), Allocation::Register(rhs)) => {
                self.build_op_reg_reg(op, lhs, rhs, out)
            }
            (Allocation::Register(lhs), Allocation::Immediate(rhs)) => {
                self.build_op_reg_imm(op, lhs, rhs, out)
            }
            (Allocation::Immediate(lhs), Allocation::Register(rhs)) => {
                self.build_op_imm_reg(op, lhs, rhs, out)
            }
            _ => panic!(),
        }
    }

    fn build_load_store_32(
        &mut self,
        op: ClauseOp32,
        short: Register,
        ext: Memory,
    ) {
        let flag = if self.was_64 { NEXT } else { 0 };
        self.was_64 = false;
        let op = op.to_u32().unwrap();
        self.out
            .push(flag | (op << 24) | ((ext.0 as u32) << 8) | (short.0 as u32));
    }
    fn build_load_ext(&mut self, src: Memory, dst: Register) {
        self.build_load_store_32(ClauseOp32::Load, dst, src);
    }

    fn build_store_ext(&mut self, src: Register, dst: Memory) {
        self.build_load_store_32(ClauseOp32::Store, src, dst);
    }

    fn build_swap_ext(&mut self, src: Register, dst: Memory) {
        self.build_load_store_32(ClauseOp32::Swap, src, dst);
    }

    /// Releases the given register
    ///
    /// Erasing the register from `self.registers` and adds it to
    /// `self.spare_short/long_registers`.
    ///
    /// Note that the caller should handle `self.allocations`
    fn release_reg(&mut self, reg: Slot) {
        self.allocations.erase_right(&reg).unwrap();
        match reg {
            Slot::Register(r) => {
                self.short_register_lru.remove(r).unwrap();
                self.spare_registers.push(r);
            }
            Slot::Memory(r) => self.spare_memory.push(r),
        }
    }

    /// Returns a short register or immediate for the given node.
    ///
    /// If the given node isn't already allocated to a short register, then we
    /// evict the oldest short register.
    fn get_allocated_value(&mut self, node: NodeIndex) -> Allocation {
        if let Some(r) = self.allocations.get_left(&node).cloned() {
            let r = match r {
                Slot::Register(r) => r,
                Slot::Memory(ext) => {
                    let out = self.swap_short_register(ext);

                    // Modify the allocations and bindings
                    let reg = Slot::Register(out);
                    let b = self.allocations.insert(node, reg);
                    assert!(b);

                    out
                }
            };
            Allocation::Register(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Allocation::Immediate(*c)
        }
    }

    /// Increment the global index counter, then drop any allocations which are
    /// no longer active (returning them to the available register list)
    fn drop_inactive_allocations(&mut self) {
        self.i += 1;

        // Drop any nodes which are no longer active
        // This could be cleaner once #62924 map_first_last is stabilized
        while let Some((index, node)) = self.active.iter().next().cloned() {
            if index >= self.i {
                break;
            }
            self.active.remove(&(index, node));

            // This node's register is now available!
            let r = self.allocations.get_left(&node).unwrap();
            self.release_reg(*r);
        }
    }

    fn update_lru(&mut self, r: Register) {
        self.short_register_lru.insert_or_update(r, self.i);
    }

    fn run(&mut self) {
        for &(n, op) in &self.t.tape {
            let out = match op {
                Op::Var(v) => {
                    let index =
                        match self.t.vars.get_by_index(v).unwrap().as_str() {
                            "X" => 0,
                            "Y" => 1,
                            _ => panic!(),
                        };
                    self.drop_inactive_allocations();
                    let out = self.get_registers();
                    // Slight misuse of the 32-bit unary form, but that's okay
                    self.build_op_reg(ClauseOp32::Input, Register(index), out);
                    out
                }
                Op::Const(c) => {
                    self.drop_inactive_allocations();
                    self.constants.insert(n, c as f32);
                    continue; // Skip the post-match logic!
                }
                Op::Binary(op, lhs, rhs) => {
                    let lhs = self.get_allocated_value(lhs);
                    if let Allocation::Register(r) = lhs {
                        self.update_lru(r);
                    }
                    let rhs = self.get_allocated_value(rhs);
                    if let Allocation::Register(r) = rhs {
                        self.update_lru(r);
                    }
                    self.drop_inactive_allocations();
                    let out = self.get_registers();
                    self.update_lru(out);
                    self.build_op(op.into(), lhs, rhs, out);
                    out
                }
                Op::BinaryChoice(op, lhs, rhs, _c) => {
                    let lhs = self.get_allocated_value(lhs);
                    if let Allocation::Register(r) = lhs {
                        self.update_lru(r);
                    }
                    let rhs = self.get_allocated_value(rhs);
                    if let Allocation::Register(r) = rhs {
                        self.update_lru(r);
                    }
                    self.drop_inactive_allocations();
                    let out = self.get_registers();
                    self.update_lru(out);
                    self.build_op(op.into(), lhs, rhs, out);
                    out
                }
                Op::Unary(op, lhs) => {
                    // Unary opcodes are only stored in short form
                    let lhs = match self.get_allocated_value(lhs) {
                        Allocation::Register(r) => r,
                        Allocation::Immediate(_i) => panic!(
                            "Unary operation on immediate should be collapsed"
                        ),
                    };
                    self.update_lru(lhs);
                    self.drop_inactive_allocations();
                    let out = self.get_registers();
                    self.update_lru(out);
                    self.build_op_reg(op.into(), lhs, out);
                    out
                }
            };
            let reg = Slot::Register(out);
            let b = self.allocations.insert(n, reg);
            assert!(b);

            self.active.insert((self.t.last_use[n], n));
            // Sanity checking
            assert_eq!(
                self.total_slots,
                self.allocations.len()
                    + self.spare_registers.len()
                    + self.spare_memory.len()
            );
        }

        // Copy from the last register to register 0
        match self.get_allocated_value(self.t.root) {
            Allocation::Immediate(f) => self.build_op_64(
                ClauseOp64::CopyImm,
                Register(0),
                f,
                Register(0),
            ),
            Allocation::Register(r) => {
                self.build_op_reg(ClauseOp32::CopyReg, r, Register(0))
            }
        }

        // Push a trailing 32-bit operation, so that we can unambiguously detect
        // if the second-to-last operation was 64-bit when walking the tape
        // backwards.
        if self.was_64 {
            self.build_op_32(
                ClauseOp32::CopyReg,
                Register(0),
                Register(0),
                Register(0),
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_interpreter() {
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let one = ctx.constant(1.0);
        let sum = ctx.add(x, one).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.to_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 2.0);
        assert_eq!(eval.f(1.0, 3.0), 2.0);
        assert_eq!(eval.f(3.0, 3.5), 3.5);
    }

    #[test]
    fn test_push() {
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.to_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.to_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.to_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.to_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.to_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.to_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0), 1.0);
    }
}
