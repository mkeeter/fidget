use std::collections::{BTreeMap, BTreeSet};

use crate::{
    compiler::{Compiler, GroupIndex, NodeIndex, Op},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    queue::PriorityQueue,
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
    fn name(&self) -> &'static str {
        match self {
            ClauseOp::Done => "DONE",
            ClauseOp::Load => "LOAD",
            ClauseOp::Store => "STORE",
            ClauseOp::Swap => "SWAP",
            ClauseOp::Input => "INPUT",
            ClauseOp::CopyReg | ClauseOp::CopyImm => "COPY",
            ClauseOp::NegReg => "NEG",
            ClauseOp::AbsReg => "ABS",
            ClauseOp::RecipReg => "RECIP",
            ClauseOp::SqrtReg => "SQRT",
            ClauseOp::SquareReg => "SQUARE",

            ClauseOp::AddRegReg | ClauseOp::AddRegImm => "ADD",
            ClauseOp::MulRegReg | ClauseOp::MulRegImm => "MUL",
            ClauseOp::SubRegReg | ClauseOp::SubImmReg | ClauseOp::SubRegImm => {
                "SUB"
            }
            ClauseOp::MinRegReg | ClauseOp::MinRegImm => "MIN",
            ClauseOp::MaxRegReg | ClauseOp::MaxRegImm => "MAX",
        }
    }
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

/// Type-safe container for an allocated (short) register
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ShortRegister(u8);

/// Type-safe container for an allocated (extended) register
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
struct ExtendedRegister(u16);

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Register {
    Short(ShortRegister),
    Extended(ExtendedRegister),
}

/// Helper `struct` to hold spare state when building an `Interpreter`
struct InterpreterBuilder<'a> {
    t: &'a Compiler,

    /// Output tape, in progress
    out: Vec<u32>,

    /// Total number of registers
    ///
    /// This should always equal the sum of `spare_short_registers`,
    /// `spare_extended_register`, and `allocations`
    register_count: usize,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_short_registers: Vec<ShortRegister>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_extended_registers: Vec<ExtendedRegister>,

    /// Active nodes, sorted by position in the input data
    active: BTreeSet<(usize, NodeIndex)>,

    /// Priority queue of recently used short registers
    ///
    /// `pop()` will return the _oldest_ short register
    short_register_lru: PriorityQueue<usize, ShortRegister>,

    /// Mapping from active nodes to available registers
    allocations: BTreeMap<NodeIndex, Register>,

    /// Mapping from allocated registers to active nodes
    registers: BTreeMap<Register, NodeIndex>,

    /// Constants declared in the compiled program
    constants: BTreeMap<NodeIndex, f32>,

    /// Position in the input data, used to determine when a node has died
    i: usize,

    /// Marks that the previous clause was 64-bit
    was_64: bool,
}

enum Allocation {
    Register(ShortRegister),
    Immediate(f32),
}

impl<'a> InterpreterBuilder<'a> {
    fn new(t: &'a Compiler) -> Self {
        Self {
            t,
            out: vec![],
            register_count: 0,
            short_register_lru: PriorityQueue::new(),
            spare_short_registers: vec![],
            spare_extended_registers: vec![],
            active: BTreeSet::new(),
            allocations: BTreeMap::new(),
            registers: BTreeMap::new(),
            constants: BTreeMap::new(),
            i: 0,
            was_64: false,
        }
    }

    /// Claims an extended register for the given node
    ///
    /// The `NodeIndex -> Register` association is updated in `self.allocations`
    fn get_extended_register(&mut self) -> ExtendedRegister {
        if let Some(r) = self.spare_extended_registers.pop() {
            r
        } else {
            let r = ExtendedRegister(self.register_count.try_into().unwrap());
            self.register_count += 1;
            r
        }
    }

    /// Claims a short register
    ///
    /// This may require evicting a short register
    fn get_short_register(&mut self) -> ShortRegister {
        let out = if let Some(r) = self.spare_short_registers.pop() {
            r
        } else if self.register_count <= u8::MAX as usize {
            let r = ShortRegister(self.register_count as u8);
            self.register_count += 1;
            r
        } else {
            // Evict a short register
            let target = self.short_register_lru.pop().unwrap();
            let node = self.registers.remove(&Register::Short(target)).unwrap();
            let ext = self.get_extended_register();
            self.build_store_ext(target, ext);
            self.registers.insert(Register::Extended(ext), node);
            self.allocations
                .entry(node)
                .and_modify(|v| *v = Register::Extended(ext));
            target
        };

        out
    }

    /// Claims a short register and loads a value from `ext`
    ///
    /// This may require evicting a short register, in which case it is swapped
    /// with `ext` (since we know `ext` is about to be available).
    fn swap_short_register(&mut self, ext: ExtendedRegister) -> ShortRegister {
        let out = if let Some(r) = self.spare_short_registers.pop() {
            self.build_load_ext(ext, r);
            r
        } else if self.register_count <= u8::MAX as usize {
            let r = ShortRegister(self.register_count as u8);
            self.register_count += 1;
            self.build_load_ext(ext, r);
            r
        } else {
            // Evict a short register
            let target = self.short_register_lru.pop().unwrap();
            let node = self.registers.remove(&Register::Short(target)).unwrap();
            self.build_swap_ext(target, ext);
            self.registers.insert(Register::Extended(ext), node);
            self.allocations
                .entry(node)
                .and_modify(|v| *v = Register::Extended(ext));
            target
        };

        out
    }

    fn build_op_32(
        &mut self,
        op: ClauseOp,
        lhs: ShortRegister,
        rhs: ShortRegister,
        out: ShortRegister,
    ) {
        let flag = if self.was_64 { 1 << 31 } else { 0 };
        self.was_64 = false;

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
        op: ClauseOp,
        arg: ShortRegister,
        imm: f32,
        out: ShortRegister,
    ) {
        let flag = if self.was_64 { 1 << 31 } else { 0 };
        self.was_64 = true;

        let op = op.to_u32().unwrap();
        assert!(op < 16384);
        self.out.push(
            flag | (1 << 30)
                | (op << 16)
                | ((arg.0 as u32) << 8)
                | (out.0 as u32),
        );
        self.out.push(imm.to_bits());
    }

    fn build_op_reg_reg(
        &mut self,
        op: ClauseOp,
        lhs: ShortRegister,
        rhs: ShortRegister,
        out: ShortRegister,
    ) {
        self.build_op_32(op, lhs, rhs, out)
    }

    fn build_op_reg(
        &mut self,
        op: ClauseOp,
        lhs: ShortRegister,
        out: ShortRegister,
    ) {
        self.build_op_32(op, lhs, ShortRegister(0), out)
    }

    fn build_op_reg_imm(
        &mut self,
        op: ClauseOp,
        lhs: ShortRegister,
        rhs: f32,
        out: ShortRegister,
    ) {
        let op = op.as_reg_imm();
        self.build_op_64(op, lhs, rhs, out);
    }

    fn build_op_imm_reg(
        &mut self,
        op: ClauseOp,
        lhs: f32,
        rhs: ShortRegister,
        out: ShortRegister,
    ) {
        let op = op.as_imm_reg();
        self.build_op_64(op, rhs, lhs, out);
    }

    fn build_op(
        &mut self,
        op: ClauseOp,
        lhs: Allocation,
        rhs: Allocation,
        out: ShortRegister,
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
        op: ClauseOp,
        short: ShortRegister,
        ext: ExtendedRegister,
    ) {
        let flag = if self.was_64 { 1 << 31 } else { 0 };
        self.was_64 = false;
        let op = op.to_u32().unwrap();
        self.out
            .push(flag | (op << 24) | ((ext.0 as u32) << 8) | (short.0 as u32));
    }
    fn build_load_ext(&mut self, src: ExtendedRegister, dst: ShortRegister) {
        self.build_load_store_32(ClauseOp::Load, dst, src);
    }

    fn build_store_ext(&mut self, src: ShortRegister, dst: ExtendedRegister) {
        self.build_load_store_32(ClauseOp::Store, src, dst);
    }

    fn build_swap_ext(&mut self, src: ShortRegister, dst: ExtendedRegister) {
        self.build_load_store_32(ClauseOp::Swap, src, dst);
    }

    /// Releases the given register
    ///
    /// Erasing the register from `self.registers` and adds it to
    /// `self.spare_short/long_registers`.
    ///
    /// Note that the caller should handle `self.allocations`
    fn release_reg(&mut self, reg: Register) {
        self.registers.remove(&reg).unwrap();
        match reg {
            Register::Short(r) => {
                self.short_register_lru.remove(r).unwrap();
                self.spare_short_registers.push(r);
            }
            Register::Extended(r) => self.spare_extended_registers.push(r),
        }
    }

    /// Returns a short register or immediate for the given node.
    ///
    /// If the given node isn't already allocated to a short register, then we
    /// evict the oldest short register.
    fn get_allocated_value(&mut self, node: NodeIndex) -> Allocation {
        if let Some(r) = self.allocations.get(&node).cloned() {
            let r = match r {
                Register::Short(r) => r,
                Register::Extended(ext) => {
                    let out = self.swap_short_register(ext);

                    // Modify the allocations and bindings
                    let reg = Register::Short(out);
                    self.registers.insert(reg, node);
                    self.allocations.insert(node, reg);

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
            let r = self.allocations.remove(&node).unwrap();
            self.release_reg(r);
        }
    }

    fn update_lru(&mut self, r: ShortRegister) {
        self.short_register_lru.insert_or_update(r, self.i);
    }

    fn recurse(&mut self, g: GroupIndex) {
        let group = &self.t.groups[g];
        for &c in &group.children {
            self.recurse(c);
        }
        for &n in &group.nodes {
            let out = match self.t.ops[n] {
                Op::Var(v) => {
                    let index =
                        match self.t.vars.get_by_index(v).unwrap().as_str() {
                            "X" => 0,
                            "Y" => 1,
                            _ => panic!(),
                        };
                    self.drop_inactive_allocations();
                    let out = self.get_short_register();
                    // Slight misuse of the 32-bit unary form, but that's okay
                    self.build_op_reg(
                        ClauseOp::Input,
                        ShortRegister(index),
                        out,
                    );
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
                    let out = self.get_short_register();
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
                    let out = self.get_short_register();
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
                    let out = self.get_short_register();
                    self.update_lru(out);
                    self.build_op_reg(op.into(), lhs, out);
                    out
                }
            };
            let reg = Register::Short(out);
            self.allocations.insert(n, reg);
            self.registers.insert(reg, n);
            self.active.insert((self.t.last_use[n], n));
        }
    }
}

impl Interpreter {
    pub fn new(t: &Compiler) -> Self {
        let mut builder = InterpreterBuilder::new(t);
        builder.recurse(t.op_group[t.root]);
        match builder.get_allocated_value(t.root) {
            Allocation::Immediate(f) => builder.build_op_64(
                ClauseOp::CopyImm,
                ShortRegister(0),
                f,
                ShortRegister(0),
            ),
            Allocation::Register(r) => {
                builder.build_op_reg(ClauseOp::CopyReg, r, ShortRegister(0))
            }
        }

        Self {
            tape: builder.out,
            registers: vec![0.0; (builder.register_count as usize).max(1)],
        }
    }

    pub fn pretty_print_tape(&self) {
        let mut iter = self.tape.iter();
        while let Some(v) = iter.next() {
            // 32-bit instruction
            if v & (1 << 30) == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp::from_u32(op).unwrap();
                match op {
                    ClauseOp::Load | ClauseOp::Store | ClauseOp::Swap => {
                        let short = v & 0xFF;
                        let ext = (v >> 8) & 0xFFFF;
                        match op {
                            ClauseOp::Load => {
                                println!("${} = LOAD ${}", short, ext)
                            }
                            ClauseOp::Store => {
                                println!("${} = STORE ${}", ext, short)
                            }
                            ClauseOp::Swap => {
                                println!("SWAP ${} ${}", short, ext)
                            }
                            _ => unreachable!(),
                        }
                    }
                    ClauseOp::Input => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let out_reg = v & 0xFF;
                        println!("${} = INPUT %{}", out_reg, lhs_reg);
                    }
                    ClauseOp::CopyReg
                    | ClauseOp::Done
                    | ClauseOp::NegReg
                    | ClauseOp::AbsReg
                    | ClauseOp::RecipReg
                    | ClauseOp::SqrtReg
                    | ClauseOp::SquareReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let out_reg = v & 0xFF;
                        println!("${} = {} ${}", out_reg, op.name(), lhs_reg);
                    }
                    ClauseOp::MaxRegReg
                    | ClauseOp::MinRegReg
                    | ClauseOp::SubRegReg
                    | ClauseOp::AddRegReg
                    | ClauseOp::MulRegReg => {
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
                    _ => panic!("Unknown 32-bit opcode"),
                }
            } else {
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp::from_u32(op).unwrap();
                let next = iter.next().unwrap();
                let imm = f32::from_bits(*next);
                let arg_reg = (v >> 8) & 0xFF;
                let out_reg = v & 0xFF;
                match op {
                    ClauseOp::MinRegImm
                    | ClauseOp::MaxRegImm
                    | ClauseOp::AddRegImm
                    | ClauseOp::MulRegImm
                    | ClauseOp::SubRegImm => {
                        println!(
                            "${} = {} ${} {}",
                            out_reg,
                            op.name(),
                            arg_reg,
                            imm
                        );
                    }
                    ClauseOp::SubImmReg => {
                        println!(
                            "${} = {} {} ${}",
                            out_reg,
                            op.name(),
                            imm,
                            arg_reg,
                        );
                    }
                    _ => panic!("Unknown 64-bit opcode"),
                }
            }
        }
        println!(
            "{} registers, {} instructions",
            self.registers.len(),
            self.tape.len()
        );
    }

    pub fn run(&mut self, x: f32, y: f32) -> f32 {
        let mut iter = self.tape.iter().enumerate();
        while let Some((_i, v)) = iter.next() {
            // 32-bit instruction
            if v & (1 << 30) == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp::from_u32(op).unwrap();
                if matches!(
                    op,
                    ClauseOp::Load | ClauseOp::Store | ClauseOp::Swap
                ) {
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
                        ClauseOp::Swap => {
                            self.registers.swap(fast_reg, extended_reg);
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
