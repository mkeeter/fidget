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
}

impl ClauseOp32 {
    fn name(&self) -> &'static str {
        match self {
            ClauseOp32::Done => "DONE",
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
    SubRegImm,
    MinRegImm,
    MaxRegImm,

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
pub struct Tape {
    pub tape: Vec<u32>,
    pub num_registers: usize,
    pub fast_reg_limit: usize,
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
        match builder.get_allocated_value(builder.t.root) {
            Allocation::Immediate(f) => builder.build_op_64(
                ClauseOp64::CopyImm,
                ShortRegister(0),
                f,
                ShortRegister(0),
            ),
            Allocation::Register(r) => {
                builder.build_op_reg(ClauseOp32::CopyReg, r, ShortRegister(0))
            }
        }

        Self {
            tape: builder.out,
            num_registers: builder.register_count as usize,
            fast_reg_limit: builder.max_fast_reg,
        }
    }

    pub fn workspace(&self) -> Vec<f32> {
        vec![0.0; self.num_registers]
    }

    pub fn eval(&self, x: f32, y: f32, workspace: &mut [f32]) -> f32 {
        let mut iter = self.tape.iter();
        while let Some(v) = iter.next() {
            // 32-bit instruction
            if v & (1 << 30) == 0 {
                let op = (v >> 24) & ((1 << 6) - 1);
                let op = ClauseOp32::from_u32(op).unwrap();
                match op {
                    ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                        let fast_reg = (v & 0xFF) as usize;
                        let extended_reg = ((v >> 8) & 0xFFFF) as usize;
                        match op {
                            ClauseOp32::Load => {
                                workspace[fast_reg] = workspace[extended_reg]
                            }
                            ClauseOp32::Store => {
                                workspace[extended_reg] = workspace[fast_reg]
                            }
                            ClauseOp32::Swap => {
                                workspace.swap(fast_reg, extended_reg);
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
                        workspace[out_reg as usize] = out;
                    }
                    ClauseOp32::CopyReg
                    | ClauseOp32::NegReg
                    | ClauseOp32::AbsReg
                    | ClauseOp32::RecipReg
                    | ClauseOp32::SqrtReg
                    | ClauseOp32::SquareReg => {
                        let lhs_reg = (v >> 16) & 0xFF;
                        let lhs = workspace[lhs_reg as usize];
                        let out_reg = v & 0xFF;
                        workspace[out_reg as usize] = match op {
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
                        let lhs = workspace[lhs_reg as usize];
                        let rhs = workspace[rhs_reg as usize];
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
                        workspace[out_reg as usize] = out;
                    }
                    _ => panic!("Bad 32-bit opcode"),
                }
            } else {
                let op = (v >> 16) & ((1 << 14) - 1);
                let op = ClauseOp64::from_u32(op).unwrap();
                let next = iter.next().unwrap();
                let arg_reg = (v >> 8) & 0xFF;
                let out_reg = v & 0xFF;
                let arg = workspace[arg_reg as usize];
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
                workspace[out_reg as usize] = out;
            }
        }
        workspace[0]
    }

    pub fn pretty_print_tape(&self) {
        let mut iter = self.tape.iter();
        while let Some(v) = iter.next() {
            // 32-bit instruction
            if v & (1 << 30) == 0 {
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
                    | ClauseOp32::Done
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
                    _ => panic!("Unknown 64-bit opcode"),
                }
            }
        }
        println!(
            "{} registers, {} instructions",
            self.num_registers,
            self.tape.len()
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

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

/// Helper `struct` to hold spare state when building an `Tape`
struct TapeBuilder<'a> {
    t: &'a Scheduled,

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
    allocations: Bimap<NodeIndex, Register>,

    /// Constants declared in the compiled program
    constants: BTreeMap<NodeIndex, f32>,

    /// Position in the input data, used to determine when a node has died
    i: usize,

    /// Marks that the previous clause was 64-bit
    was_64: bool,

    /// The highest index of a fast register
    ///
    /// When used by the interpreter, this is 255 (since fast registers are
    /// packed into 8 bits); however, it can be reduced if we're compiling down
    /// to assembly.
    max_fast_reg: usize,
}

enum Allocation {
    Register(ShortRegister),
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
            register_count: 0,
            short_register_lru: PriorityQueue::new(),
            spare_short_registers: vec![],
            spare_extended_registers: vec![],
            active: BTreeSet::new(),
            allocations: Bimap::new(),
            constants: BTreeMap::new(),
            i: 0,
            was_64: false,
            max_fast_reg: reg_limit,
        }
    }

    /// Claims an extended register for the given node
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
        if let Some(r) = self.spare_short_registers.pop() {
            r
        } else if self.register_count <= self.max_fast_reg {
            let r = ShortRegister(self.register_count as u8);
            self.register_count += 1;
            r
        } else {
            // Evict a short register
            let target = self.short_register_lru.pop().unwrap();
            let node = self
                .allocations
                .erase_right(&Register::Short(target))
                .unwrap();
            let ext = self.get_extended_register();
            self.build_store_ext(target, ext);
            let b = self.allocations.insert(node, Register::Extended(ext));
            assert!(b);
            target
        }
    }

    /// Claims a short register and loads a value from `ext`
    ///
    /// This may require evicting a short register, in which case it is swapped
    /// with `ext` (since we know `ext` is about to be available).
    fn swap_short_register(&mut self, ext: ExtendedRegister) -> ShortRegister {
        if let Some(r) = self.spare_short_registers.pop() {
            assert!(self.allocations.get_right(&Register::Short(r)).is_none());
            self.build_load_ext(ext, r);
            self.release_reg(Register::Extended(ext));
            r
        } else if self.register_count <= self.max_fast_reg {
            let r = ShortRegister(self.register_count as u8);
            self.register_count += 1;
            self.build_load_ext(ext, r);
            self.release_reg(Register::Extended(ext));
            r
        } else {
            // Evict a short register, reassigning its allocation data
            let target = self.short_register_lru.pop().unwrap();
            let node = self
                .allocations
                .erase_right(&Register::Short(target))
                .unwrap();
            self.build_swap_ext(target, ext);
            self.allocations
                .erase_right(&Register::Extended(ext))
                .unwrap();
            let b = self.allocations.insert(node, Register::Extended(ext));
            assert!(b);
            target
        }
    }

    fn build_op_32(
        &mut self,
        op: ClauseOp32,
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
        op: ClauseOp64,
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
        op: ClauseOp32,
        lhs: ShortRegister,
        rhs: ShortRegister,
        out: ShortRegister,
    ) {
        self.build_op_32(op, lhs, rhs, out)
    }

    fn build_op_reg(
        &mut self,
        op: ClauseOp32,
        lhs: ShortRegister,
        out: ShortRegister,
    ) {
        self.build_op_32(op, lhs, ShortRegister(0), out)
    }

    fn build_op_reg_imm(
        &mut self,
        op: ClauseOp32,
        lhs: ShortRegister,
        rhs: f32,
        out: ShortRegister,
    ) {
        let op = op.as_reg_imm();
        self.build_op_64(op, lhs, rhs, out);
    }

    fn build_op_imm_reg(
        &mut self,
        op: ClauseOp32,
        lhs: f32,
        rhs: ShortRegister,
        out: ShortRegister,
    ) {
        let op = op.as_imm_reg();
        self.build_op_64(op, rhs, lhs, out);
    }

    fn build_op(
        &mut self,
        op: ClauseOp32,
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
        op: ClauseOp32,
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
        self.build_load_store_32(ClauseOp32::Load, dst, src);
    }

    fn build_store_ext(&mut self, src: ShortRegister, dst: ExtendedRegister) {
        self.build_load_store_32(ClauseOp32::Store, src, dst);
    }

    fn build_swap_ext(&mut self, src: ShortRegister, dst: ExtendedRegister) {
        self.build_load_store_32(ClauseOp32::Swap, src, dst);
    }

    /// Releases the given register
    ///
    /// Erasing the register from `self.registers` and adds it to
    /// `self.spare_short/long_registers`.
    ///
    /// Note that the caller should handle `self.allocations`
    fn release_reg(&mut self, reg: Register) {
        self.allocations.erase_right(&reg).unwrap();
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
        if let Some(r) = self.allocations.get_left(&node).cloned() {
            let r = match r {
                Register::Short(r) => r,
                Register::Extended(ext) => {
                    let out = self.swap_short_register(ext);

                    // Modify the allocations and bindings
                    let reg = Register::Short(out);
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

    fn update_lru(&mut self, r: ShortRegister) {
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
                    let out = self.get_short_register();
                    // Slight misuse of the 32-bit unary form, but that's okay
                    self.build_op_reg(
                        ClauseOp32::Input,
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
            let b = self.allocations.insert(n, reg);
            assert!(b);

            self.active.insert((self.t.last_use[n], n));
            // Sanity checking
            assert_eq!(
                self.register_count,
                self.allocations.len()
                    + self.spare_short_registers.len()
                    + self.spare_extended_registers.len()
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
        let mut workspace = tape.workspace();
        assert_eq!(tape.eval(1.0, 2.0, &mut workspace), 2.0);
        assert_eq!(tape.eval(1.0, 3.0, &mut workspace), 2.0);
        assert_eq!(tape.eval(3.0, 3.5, &mut workspace), 3.5);
    }
}
