use std::collections::{BTreeMap, BTreeSet};

use crate::{
    compiler::{Compiler, GroupIndex, NodeIndex, Op},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
};

use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Copy, Clone, Debug, ToPrimitive, FromPrimitive)]
pub enum ClauseOp {
    // ------------ 32-bit opcodes --------------
    /// `Done` marks the end of the tape.  It is only needed so that we can
    /// label a preceeding 64-bit operation (in the high bit).
    Done = 0,

    /// Load from an extended register to a main register
    Load,

    /// Store from a main register to an extended register
    Store,

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

    // ------------ 64-bit opcodes --------------
    /// Add a register and an immediate
    AddRegImm = 64,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Subtract a register from an immediate
    SubImmReg,
    SubRegImm,
    MinRegImm,
    MaxRegImm,

    CopyImm,
}

impl ClauseOp {
    fn as_reg_imm(&self) -> Self {
        match self {
            ClauseOp::AddRegReg => ClauseOp::AddRegImm,
            ClauseOp::SubRegReg => ClauseOp::SubRegImm,
            ClauseOp::MulRegReg => ClauseOp::MulRegImm,
            ClauseOp::MinRegReg => ClauseOp::MinRegImm,
            ClauseOp::MaxRegReg => ClauseOp::MaxRegImm,
            _ => panic!(),
        }
    }
    fn as_imm_reg(&self) -> Self {
        match self {
            ClauseOp::AddRegReg => ClauseOp::AddRegImm,
            ClauseOp::SubRegReg => ClauseOp::SubImmReg,
            ClauseOp::MulRegReg => ClauseOp::MulRegImm,
            ClauseOp::MinRegReg => ClauseOp::MinRegImm,
            ClauseOp::MaxRegReg => ClauseOp::MaxRegImm,
            _ => panic!(),
        }
    }
}

impl From<BinaryOpcode> for ClauseOp {
    fn from(b: BinaryOpcode) -> Self {
        match b {
            BinaryOpcode::Sub => ClauseOp::SubRegReg,
            BinaryOpcode::Add => ClauseOp::AddRegReg,
            BinaryOpcode::Mul => ClauseOp::MulRegReg,
        }
    }
}

impl From<BinaryChoiceOpcode> for ClauseOp {
    fn from(b: BinaryChoiceOpcode) -> Self {
        match b {
            BinaryChoiceOpcode::Min => ClauseOp::MinRegReg,
            BinaryChoiceOpcode::Max => ClauseOp::MaxRegReg,
        }
    }
}

impl From<UnaryOpcode> for ClauseOp {
    fn from(b: UnaryOpcode) -> Self {
        match b {
            UnaryOpcode::Square => ClauseOp::SquareReg,
            UnaryOpcode::Sqrt => ClauseOp::SqrtReg,
            UnaryOpcode::Abs => ClauseOp::AbsReg,
            UnaryOpcode::Recip => ClauseOp::RecipReg,
            UnaryOpcode::Neg => ClauseOp::NegReg,
        }
    }
}

/// The instruction tape is given as a `Vec<u32>`.
///
/// Decoding depends on the upper bits of a particular `u32`:
///
/// | Bits | Meaning                        |
/// |------|--------------------------------|
/// | 31   | Previous instruction is 64-bit |
/// | 30   | This instruction is 64-bit     |
///
/// Bit 31 allows us to read the tape in either direction, as long as the final
/// item in the tape is a 32-bit operation.
///
/// Lower bits depend on bit 30.
///
/// # 32-bit instructions
/// ## Common operations on fast registers
/// For a 32-bit instruction (i.e. bit 30 is 0), there are two encodings.  The
/// most common is a operation on fast registers:
///
/// | Bits  | Meaning |
/// |-------|---------|
/// | 29-24 | Opcode  |
/// | 23-16 | LHS     |
/// | 16-8  | RHS     |
/// | 7-0   | Out     |
///
/// This allows us to encode the 64 most common opcodes into a single `u32`.
///
/// ## Load and store
/// The common operation encoding limits us to the lower 256 registers; if they
/// need to use a different register, then they can use Load and Store
/// instructions to shuffle data around.  This is an alternate form of a 32-bit
/// operation:
///
/// | Bits  | Meaning           |
/// |-------|-------------------|
/// | 29-24 | `Load` or `Store` |
/// | 23-8  | Extended register |
/// | 7-0   | Fast register     |
///
/// # 64-bit instructions
/// ## Immediate form
/// | Bits  | Meaning       |
/// |-------|---------------|
/// | 29-16 | Clause opcode |
/// | 16-8  | Arg1          |
/// | 7-0   | Out           |
///
/// Followed by an immediate (as a `f32` as raw bits)
///
/// ## Uncommon operation
/// TBD
#[derive(Debug)]
pub struct Interpreter {
    tape: Vec<u32>,

    /// Working memory for registers.
    registers: Vec<f32>,
}

/// Type-safe container for an allocated register
#[derive(Copy, Clone, Debug)]
struct Register(u32);

/// Helper `struct` to hold spare state when building an `Interpreter`
struct InterpreterBuilder<'a> {
    t: &'a Compiler,

    /// Output tape, in progress
    out: Vec<u32>,

    /// Total number of registers
    ///
    /// This should always equal the sum of `spare_registers` and `allocations`
    register_count: u32,

    /// Available registers, with the most recently available at the back
    spare_registers: Vec<Register>,

    /// Active nodes, sorted by position in the input data
    active: BTreeSet<(usize, NodeIndex)>,

    /// Mapping from active nodes to available registers
    allocations: BTreeMap<NodeIndex, Register>,

    /// Constants declared in the compiled program
    constants: BTreeMap<NodeIndex, f32>,

    /// Position in the input data, used to determine when a node has died
    i: usize,

    /// Marks that the previous clause was 64-bit
    was_64: bool,
}

enum Allocation {
    Register(Register),
    Immediate(f32),
}

impl<'a> InterpreterBuilder<'a> {
    fn new(t: &'a Compiler) -> Self {
        Self {
            t,
            out: vec![],
            register_count: 0,
            spare_registers: vec![],
            active: BTreeSet::new(),
            allocations: BTreeMap::new(),
            constants: BTreeMap::new(),
            i: 0,
            was_64: false,
        }
    }

    /// Claims a register for the given node
    ///
    /// The `NodeIndex -> Register` association is saved in `self.allocations`,
    /// and the node is marked as active in `self.active`
    fn claim_short_register(&mut self, n: NodeIndex) -> Register {
        let out = if let Some(r) = self.spare_registers.pop() {
            r
        } else {
            let r = Register(self.register_count);
            self.register_count += 1;
            r
        };
        self.allocations.insert(n, out);

        // This node is now active!
        self.active.insert((self.t.last_use[n], n));

        out
    }

    fn push_op_short(
        &mut self,
        op: ClauseOp,
        lhs: Register,
        rhs: Register,
        out: Register,
    ) {
        let flag = if self.was_64 { 1 << 31 } else { 0 };
        self.was_64 = false;

        let op = op.to_u8().unwrap() as u32;
        assert!(op < 64);
        assert!(lhs.0 < 256);
        assert!(rhs.0 < 256);
        assert!(out.0 < 256);
        self.out
            .push(flag | (op << 24) | (lhs.0 << 16) | (rhs.0 << 8) | out.0);
    }

    fn push_op_long(
        &mut self,
        op: ClauseOp,
        arg: Register,
        imm: f32,
        out: Register,
    ) {
        let flag = if self.was_64 { 1 << 31 } else { 0 };
        self.was_64 = true;

        let op = op.to_u8().unwrap() as u32;
        assert!(op < 16384);
        assert!(arg.0 < 256);
        assert!(out.0 < 256);
        self.out
            .push(flag | (1 << 30) | (op << 16) | (arg.0 << 8) | (out.0));
        self.out.push(imm.to_bits());
    }

    fn push_op_reg_reg(
        &mut self,
        op: ClauseOp,
        lhs: Register,
        rhs: Register,
        out: Register,
    ) {
        self.push_op_short(op, lhs, rhs, out)
    }

    fn push_op_reg(&mut self, op: ClauseOp, lhs: Register, out: Register) {
        self.push_op_short(op, lhs, Register(0), out)
    }

    fn push_op_reg_imm(
        &mut self,
        op: ClauseOp,
        lhs: Register,
        rhs: f32,
        out: Register,
    ) {
        let op = op.as_reg_imm();
        self.push_op_long(op, lhs, rhs, out);
    }

    fn push_op_imm_reg(
        &mut self,
        op: ClauseOp,
        lhs: f32,
        rhs: Register,
        out: Register,
    ) {
        let op = op.as_imm_reg();
        self.push_op_long(op, rhs, lhs, out);
    }

    fn push_op(
        &mut self,
        op: ClauseOp,
        lhs: Allocation,
        rhs: Allocation,
        out: Register,
    ) {
        match (lhs, rhs) {
            (Allocation::Register(lhs), Allocation::Register(rhs)) => {
                self.push_op_reg_reg(op, lhs, rhs, out)
            }
            (Allocation::Register(lhs), Allocation::Immediate(rhs)) => {
                self.push_op_reg_imm(op, lhs, rhs, out)
            }
            (Allocation::Immediate(lhs), Allocation::Register(rhs)) => {
                self.push_op_imm_reg(op, lhs, rhs, out)
            }
            _ => panic!(),
        }
    }

    fn get_allocated_value(&self, node: NodeIndex) -> Allocation {
        if let Some(r) = self.allocations.get(&node) {
            Allocation::Register(*r)
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
            self.spare_registers
                .push(self.allocations.remove(&node).unwrap());
        }
    }

    fn recurse(&mut self, g: GroupIndex) {
        let group = &self.t.groups[g];
        for &c in &group.children {
            self.recurse(c);
        }
        for &n in &group.nodes {
            match self.t.ops[n] {
                Op::Var(v) => {
                    let index =
                        match self.t.vars.get_by_index(v).unwrap().as_str() {
                            "X" => 0,
                            "Y" => 1,
                            _ => panic!(),
                        };
                    self.drop_inactive_allocations();
                    let out = self.claim_short_register(n);
                    // Slight misuse of the 32-bit unary form, but that's okay
                    self.push_op_reg(ClauseOp::Input, Register(index), out);
                }
                Op::Const(c) => {
                    self.drop_inactive_allocations();
                    self.constants.insert(n, c as f32);
                }
                Op::Binary(op, lhs, rhs) => {
                    let lhs = self.get_allocated_value(lhs);
                    let rhs = self.get_allocated_value(rhs);
                    self.drop_inactive_allocations();
                    let out = self.claim_short_register(n);
                    self.push_op(op.into(), lhs, rhs, out)
                }
                Op::BinaryChoice(op, lhs, rhs, _c) => {
                    let lhs = self.get_allocated_value(lhs);
                    let rhs = self.get_allocated_value(rhs);
                    self.drop_inactive_allocations();
                    let out = self.claim_short_register(n);
                    self.push_op(op.into(), lhs, rhs, out)
                }
                Op::Unary(op, lhs) => {
                    // Unary opcodes are only stored in short form
                    let lhs = *self.allocations.get(&lhs).unwrap();
                    self.drop_inactive_allocations();
                    let out = self.claim_short_register(n);
                    self.push_op_reg(op.into(), lhs, out)
                }
            }
        }
    }
}

impl Interpreter {
    pub fn new(t: &Compiler) -> Self {
        let mut builder = InterpreterBuilder::new(t);
        builder.recurse(t.op_group[t.root]);
        match builder.get_allocated_value(t.root) {
            Allocation::Immediate(f) => builder.push_op_long(
                ClauseOp::CopyImm,
                Register(0),
                f,
                Register(0),
            ),
            Allocation::Register(r) => {
                builder.push_op_reg(ClauseOp::CopyReg, r, Register(0))
            }
        }

        Self {
            tape: builder.out,
            registers: vec![0.0; (builder.register_count as usize).max(1)],
        }
    }

    pub fn run(&mut self, x: f32, y: f32) -> f32 {
        let mut iter = self.tape.iter().enumerate();
        while let Some((_i, v)) = iter.next() {
            // 32-bit instruction
            if v & (1 << 30) == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp::from_u32(op).unwrap();
                if matches!(op, ClauseOp::Load | ClauseOp::Store) {
                    let fast_reg = (v & 0xFF) as usize;
                    let extended_reg = ((v >> 8) & 0xFFFF) as usize;
                    match op {
                        ClauseOp::Load => {
                            self.registers[fast_reg] =
                                self.registers[extended_reg]
                        }
                        ClauseOp::Store => {
                            self.registers[extended_reg] =
                                self.registers[fast_reg]
                        }
                        _ => unreachable!(),
                    }
                } else {
                    let lhs_reg = (v >> 16) & 0xFF;
                    let rhs_reg = (v >> 8) & 0xFF;
                    let out_reg = v & 0xFF;
                    let lhs = self.registers[lhs_reg as usize];
                    let rhs = self.registers[rhs_reg as usize];
                    let out = match op {
                        ClauseOp::Input => match lhs_reg {
                            0 => x,
                            1 => y,
                            _ => panic!(),
                        },
                        ClauseOp::CopyReg => lhs,
                        ClauseOp::NegReg => -lhs,
                        ClauseOp::AbsReg => lhs.abs(),
                        ClauseOp::RecipReg => 1.0 / lhs,
                        ClauseOp::SqrtReg => lhs.sqrt(),
                        ClauseOp::SquareReg => lhs * lhs,
                        ClauseOp::AddRegReg => lhs + rhs,
                        ClauseOp::MulRegReg => lhs * rhs,
                        ClauseOp::SubRegReg => lhs - rhs,
                        ClauseOp::MinRegReg => lhs.min(rhs),
                        ClauseOp::MaxRegReg => lhs.max(rhs),
                        ClauseOp::Load | ClauseOp::Store => unreachable!(),
                        _ => panic!(),
                    };
                    self.registers[out_reg as usize] = out;
                }
            } else {
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp::from_u32(op).unwrap();
                let (_j, next) = iter.next().unwrap();
                let arg_reg = (v >> 8) & 0xFF;
                let out_reg = v & 0xFF;
                let arg = self.registers[arg_reg as usize];
                let imm = f32::from_bits(*next);
                let out = match op {
                    ClauseOp::AddRegImm => arg + imm,
                    ClauseOp::MulRegImm => arg * imm,
                    ClauseOp::SubImmReg => imm - arg,
                    ClauseOp::SubRegImm => arg - imm,
                    ClauseOp::MinRegImm => arg.min(imm),
                    ClauseOp::MaxRegImm => arg.max(imm),
                    ClauseOp::CopyImm => imm,
                    _ => panic!(),
                };
                self.registers[out_reg as usize] = out;
            }
        }
        self.registers[0]
    }
}

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
        let compiled = Compiler::new(&ctx, min);
        let mut interp = Interpreter::new(&compiled);
        assert_eq!(interp.run(1.0, 2.0), 2.0);
        assert_eq!(interp.run(1.0, 3.0), 2.0);
        assert_eq!(interp.run(3.0, 3.5), 3.5);

        let compiled = Compiler::new(&ctx, one);
        let mut interp = Interpreter::new(&compiled);
        assert_eq!(interp.run(1.0, 2.0), 1.0);
        assert_eq!(interp.run(2.0, 3.0), 1.0);
    }
}
