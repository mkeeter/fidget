use crate::compiler::{Lru, RegOp, RegTape, SsaOp};

use arrayvec::ArrayVec;

#[derive(Copy, Clone, Debug)]
enum Allocation {
    Register(u8),
    Memory(u32),
    Unassigned,
}

/// Cheap and cheerful single-pass register allocation
pub struct RegisterAllocator {
    /// Map from the index in the original (globally allocated) tape to a
    /// specific register or memory slot.
    allocations: Vec<u32>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `u32::MAX` if the register is currently unused.
    ///
    /// The inner `u32` here is an index into the original (SSA) tape
    ///
    /// Only the first `reg_limit` indexes are valid, but we use a fixed
    /// (maximum) size array for speed.
    registers: [u32; u8::MAX as usize],

    /// Stores a least-recently-used list of register
    ///
    /// This is sized with a backing array that can hold the maximum register
    /// count (`u8::MAX`), but will be constructed with the specific register
    /// limit in `new()`.
    register_lru: Lru,

    /// User-defined register limit; beyond this point we use load/store
    /// operations to move values to and from memory.
    reg_limit: u8,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back
    spare_registers: ArrayVec<u8, { u8::MAX as usize }>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_memory: Vec<u32>,

    /// Output slots, assembled in reverse order
    out: RegTape,
}

impl RegisterAllocator {
    /// Builds a new `RegisterAllocator`.
    ///
    /// Upon construction, SSA register 0 is bound to local register 0; you
    /// would be well advised to use it as the output of your function.
    pub fn new(reg_limit: u8, size: usize) -> Self {
        let mut out = Self {
            allocations: vec![u32::MAX; size],

            registers: [u32::MAX; u8::MAX as usize],
            register_lru: Lru::new(reg_limit),

            reg_limit,
            spare_registers: ArrayVec::new(),
            spare_memory: Vec::with_capacity(1024),

            out: RegTape::empty(),
        };
        out.bind_register(0, 0);
        out
    }

    /// Build a new empty register allocator
    pub fn empty() -> Self {
        Self {
            allocations: vec![],

            registers: [u32::MAX; u8::MAX as usize],
            register_lru: Lru::new(0),

            reg_limit: 0,
            spare_registers: ArrayVec::new(),
            spare_memory: vec![],

            out: RegTape::empty(),
        }
    }

    /// Resets internal state, reusing allocations and the provided tape
    pub fn reset(&mut self, reg_limit: u8, size: usize, tape: RegTape) {
        assert!(self.out.is_empty());
        self.allocations.fill(u32::MAX);
        self.allocations.resize(size, u32::MAX);
        self.registers.fill(u32::MAX);
        self.register_lru = Lru::new(reg_limit);
        self.reg_limit = reg_limit;
        self.spare_registers.clear();
        self.spare_memory.clear();
        self.out = tape;
        self.out.reset();
        self.bind_register(0, 0);
    }

    /// Claims the internal `Vec<RegOp>`, leaving it empty
    #[inline]
    pub fn finalize(&mut self) -> RegTape {
        std::mem::take(&mut self.out)
    }

    /// Returns an available memory slot.
    ///
    /// Memory is treated as unlimited; if we don't have any spare slots, then
    /// we'll assign a new one (incrementing `self.total_slots`).
    ///
    /// > If there's one thing I love  
    /// > It's an infinite resource  
    /// > If there's one thing worth loving  
    /// > It's a surplus of supplies
    #[inline]
    fn get_memory(&mut self) -> u32 {
        if let Some(p) = self.spare_memory.pop() {
            p
        } else {
            let out = self.out.slot_count;
            self.out.slot_count += 1;
            assert!(out >= self.reg_limit.into());
            out
        }
    }

    /// Finds the oldest register
    ///
    /// This is useful when deciding which register to evict to make room
    #[inline]
    fn oldest_reg(&mut self) -> u8 {
        self.register_lru.pop()
    }

    /// Returns the slot allocated to the given node
    ///
    /// The input is an SSA assignment (i.e. an assignment in the input
    /// `SsaTape`)
    ///
    /// If the output is a register, then it's poked to update recency
    #[inline]
    fn get_allocation(&mut self, n: u32) -> Allocation {
        match self.allocations[n as usize] {
            i if i < self.reg_limit as u32 => {
                self.register_lru.poke(i as u8);
                Allocation::Register(i as u8)
            }
            u32::MAX => Allocation::Unassigned,
            i => Allocation::Memory(i),
        }
    }

    /// Return an unoccupied register, if available
    #[inline]
    fn get_spare_register(&mut self) -> Option<u8> {
        self.spare_registers.pop().or_else(|| {
            if self.out.slot_count < self.reg_limit as u32 {
                let reg = self.out.slot_count;
                assert!(self.registers[reg as usize] == u32::MAX);
                self.out.slot_count += 1;
                Some(reg.try_into().unwrap())
            } else {
                None
            }
        })
    }

    #[inline]
    fn get_register(&mut self) -> u8 {
        if let Some(reg) = self.get_spare_register() {
            assert_eq!(self.registers[reg as usize], u32::MAX);
            self.register_lru.poke(reg);
            reg
        } else {
            // Slot is in memory, and no spare register is available
            let reg = self.oldest_reg();

            // Here's where it will go:
            let mem = self.get_memory();

            // Whoever was previously using you is in for a surprise
            let prev_node = self.registers[reg as usize];
            self.allocations[prev_node as usize] = mem;

            // This register is now unassigned
            self.registers[reg as usize] = u32::MAX;

            self.out.push(RegOp::Load(reg, mem));
            reg
        }
    }

    #[inline]
    fn rebind_register(&mut self, n: u32, reg: u8) {
        assert!(self.allocations[n as usize] >= self.reg_limit as u32);
        assert!(self.registers[reg as usize] != u32::MAX);

        let prev_node = self.registers[reg as usize];
        self.allocations[prev_node as usize] = u32::MAX;

        // Bind the register and update its use time
        self.registers[reg as usize] = n;
        self.allocations[n as usize] = reg as u32;
        self.register_lru.poke(reg);
    }

    #[inline]
    fn bind_register(&mut self, n: u32, reg: u8) {
        assert!(self.allocations[n as usize] >= self.reg_limit as u32);
        assert!(self.registers[reg as usize] == u32::MAX);

        // Bind the register and update its use time
        self.registers[reg as usize] = n;
        self.allocations[n as usize] = reg as u32;
        self.register_lru.poke(reg);
    }

    /// Release a register back to the pool of spares
    #[inline]
    fn release_reg(&mut self, reg: u8) {
        // Release the output register, so it could be used for inputs
        assert!(reg < self.reg_limit);

        let node = self.registers[reg as usize];
        assert!(node != u32::MAX);

        self.registers[reg as usize] = u32::MAX;
        self.spare_registers.push(reg);
        // Modifying self.allocations isn't strictly necessary, but could help
        // us detect logical errors (since it should never be used after this)
        self.allocations[node as usize] = u32::MAX;
    }

    #[inline]
    fn release_mem(&mut self, mem: u32) {
        assert!(mem >= self.reg_limit as u32);
        self.spare_memory.push(mem);
        // This leaves self.allocations[...] stil pointing to the memory slot,
        // but that's okay, because it should never be used
    }

    /// Lowers an operation that uses a single register into an
    /// [`RegOp`], pushing it to the internal tape.
    ///
    /// This may also push `Load` or `Store` instructions to the internal tape,
    /// if there aren't enough spare registers.
    #[inline(always)]
    fn op_reg(&mut self, op: SsaOp) {
        let (out, arg, op): (u32, u32, fn(u8, u8) -> RegOp) = match op {
            SsaOp::NegReg(out, arg) => (out, arg, RegOp::NegReg),
            SsaOp::AbsReg(out, arg) => (out, arg, RegOp::AbsReg),
            SsaOp::RecipReg(out, arg) => (out, arg, RegOp::RecipReg),
            SsaOp::SqrtReg(out, arg) => (out, arg, RegOp::SqrtReg),
            SsaOp::SquareReg(out, arg) => (out, arg, RegOp::SquareReg),
            SsaOp::CopyReg(out, arg) => (out, arg, RegOp::CopyReg),
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, op);
    }

    /// Allocates the next operation in the tape
    #[inline(always)]
    pub fn op(&mut self, op: SsaOp) {
        match op {
            SsaOp::Var(out, i) => self.op_var(out, i),
            SsaOp::Input(out, i) => self.op_input(out, i.try_into().unwrap()),
            SsaOp::CopyImm(out, imm) => self.op_copy_imm(out, imm),

            SsaOp::NegReg(..)
            | SsaOp::AbsReg(..)
            | SsaOp::RecipReg(..)
            | SsaOp::SqrtReg(..)
            | SsaOp::SquareReg(..)
            | SsaOp::CopyReg(..) => self.op_reg(op),

            SsaOp::AddRegImm(..)
            | SsaOp::SubRegImm(..)
            | SsaOp::SubImmReg(..)
            | SsaOp::MulRegImm(..)
            | SsaOp::DivRegImm(..)
            | SsaOp::DivImmReg(..)
            | SsaOp::MinRegImm(..)
            | SsaOp::MaxRegImm(..) => self.op_reg_imm(op),

            SsaOp::AddRegReg(..)
            | SsaOp::SubRegReg(..)
            | SsaOp::MulRegReg(..)
            | SsaOp::DivRegReg(..)
            | SsaOp::MinRegReg(..)
            | SsaOp::MaxRegReg(..) => self.op_reg_reg(op),
        }
    }

    fn push_store(&mut self, reg: u8, mem: u32) {
        self.out.push(RegOp::Store(reg, mem));
        self.release_mem(mem);
    }

    /// Returns a register that is bound to the given SSA input
    ///
    /// If the given SSA input is not already bound to a register, then we
    /// evict the oldest register using `Self::get_register`, with the
    /// appropriate set of LOAD/STORE operations.
    #[inline]
    fn get_out_reg(&mut self, out: u32) -> u8 {
        match self.get_allocation(out) {
            Allocation::Register(r_x) => r_x,
            Allocation::Memory(m_x) => {
                // TODO: this could be more efficient with a Swap instruction,
                // since we know that we're about to free a memory slot.
                let r_a = self.get_register();

                self.push_store(r_a, m_x);
                self.bind_register(out, r_a);
                r_a
            }
            Allocation::Unassigned => panic!("Cannot have unassigned output"),
        }
    }

    #[inline(always)]
    fn op_reg_fn(&mut self, out: u32, arg: u32, op: impl Fn(u8, u8) -> RegOp) {
        // When we enter this function, the output can be assigned to either a
        // register or memory, and the input can be a register, memory, or
        // unassigned.  This gives us six unique situations.
        //
        //   out | arg | what do?
        //  ================================================================
        //   r_x | r_y | r_x = op r_y
        //       |     |
        //       |     | Afterwards, r_x is free
        //  -----|-----|----------------------------------------------------
        //   r_x | m_y | r_x = op r_a
        //       |     | store r_a -> m_y
        //       |     | [load r_a <- m_a]
        //       |     |
        //       |     | Afterward, r_x is free and r_a points to the former m_y
        //  -----|-----|----------------------------------------------------
        //   r_x |  U  | r_x = op r_x
        //       |     |
        //       |     | Afterward, r_x points to the arg
        //  -----|-----|----------------------------------------------------
        //
        //  Cases with the output in memory (m_x) are identical except that they
        //  include a trailing
        //
        //      store r_a -> m_x
        //      [load r_a <- m_a]
        //
        // i.e. storing the value in the assigned memory slot, and then
        // restoring the previous register value if present (when read forward).
        let r_x = self.get_out_reg(out);
        match self.get_allocation(arg) {
            Allocation::Register(r_y) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            Allocation::Memory(m_y) => {
                let r_a = self.get_register();
                self.push_store(r_a, m_y);
                self.out.push(op(r_x, r_a));
                self.release_reg(r_x);
                self.bind_register(arg, r_a);
            }
            Allocation::Unassigned => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);
            }
        }
    }

    /// Lowers a two-register operation into an [`RegOp`], pushing it to the
    /// internal tape.
    ///
    /// Inputs are SSA registers from a [`SsaTape`](crate::compiler::SsaTape),
    /// i.e. globally addressed.
    ///
    /// If there aren't enough spare registers, this may also push `Load` or
    /// `Store` instructions to the internal tape.  It's trickier than it
    /// sounds; look at the source code for a table showing all 18 (!) possible
    /// configurations.
    #[inline(always)]
    fn op_reg_reg(&mut self, op: SsaOp) {
        // Looking at this horrific table, you may be tempted to think "surely
        // there's a clean abstraction that wraps this up in a few functions".
        // You may be right, but I spent a few days chasing down terrible memory
        // load/store ordering bugs, and decided that the brute-force approach
        // was the right one.
        //
        //   out | lhs  | rhs  | what do?
        //  ================================================================
        //  r_x  | r_y  | r_z  | r_x = op r_y r_z
        //       |      |      |
        //       |      |      | Afterwards, r_x is free
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | r_z  | r_x = op r_a r_z
        //       |      |      | store r_a -> m_y
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_x is free, r_a points to the
        //       |      |      | former m_y, and m_y is free
        //  -----|------|------|----------------------------------------------
        //  r_x  | r_y  | m_z  | ibid
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | m_z  | r_x = op r_a r_b
        //       |      |      | store r_b -> m_z
        //       |      |      | [load r_b <- m_b]
        //       |      |      | store r_a -> m_y
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_x points to the former m_y, r_a
        //       |      |      | points to the former m_z, m_y and m_z are free,
        //       |      |      | [and m_a points to the former r_a]
        //  -----|------|------|----------------------------------------------
        //  r_x  | U    | r_z  | r_x = op r_x r_z
        //       |      |      |
        //       |      |      | Afterward, r_x points to the lhs
        //  -----|------|------|----------------------------------------------
        //  r_x  | r_y  | U    | ibid
        //  -----|------|------|----------------------------------------------
        //  r_x  | U    | U    | rx = op r_x r_a
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterward, r_x points to the lhs, r_a points to
        //       |      |      | rhs, [and m_a points to the former r_a]
        //  -----|------|------|----------------------------------------------
        //  r_x  | U    | m_z  | r_x = op r_x r_a
        //       |      |      | store r_a -> m_z
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterward, r_x points to the lhs, r_a points to
        //       |      |      | rhs, m_z is free, [and m_a points to the former
        //       |      |      | r_a]
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | U    | ibid
        //  =====|======|======|==============================================
        //
        //  The operations with the output in the memory slot are identical,
        //  except that they end with
        //      store r_o -> m_o
        //      [load r_o <- m_o]
        //  (i.e. moving the register to memory immediately, and optionally
        //  restoring the previous value.  Here's an example:
        //
        //  -----|------|------|----------------------------------------------
        //   m_x | r_y  | r_z  | r_a = op r_y r_z
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_a and m_x are free, [m_a points
        //       |      |      | to the former r_a}
        //  -----|------|------|----------------------------------------------
        let (out, lhs, rhs, op): (_, _, _, fn(u8, u8, u8) -> RegOp) = match op {
            SsaOp::AddRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::AddRegReg)
            }
            SsaOp::SubRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::SubRegReg)
            }
            SsaOp::MulRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::MulRegReg)
            }
            SsaOp::DivRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::DivRegReg)
            }
            SsaOp::MinRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::MinRegReg)
            }
            SsaOp::MaxRegReg(out, lhs, rhs) => {
                (out, lhs, rhs, RegOp::MaxRegReg)
            }
            _ => panic!("Bad opcode: {op:?}"),
        };
        let r_x = self.get_out_reg(out);
        match (self.get_allocation(lhs), self.get_allocation(rhs)) {
            (Allocation::Register(r_y), Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (Allocation::Memory(m_y), Allocation::Register(r_z)) => {
                let r_a = self.get_register();
                self.push_store(r_a, m_y);
                self.out.push(op(r_x, r_a, r_z));
                self.release_reg(r_x);
                self.bind_register(lhs, r_a);
            }
            (Allocation::Register(r_y), Allocation::Memory(m_z)) => {
                let r_a = self.get_register();
                self.push_store(r_a, m_z);
                self.out.push(op(r_x, r_y, r_a));
                self.release_reg(r_x);
                self.bind_register(rhs, r_a);
            }
            (Allocation::Memory(m_y), Allocation::Memory(..)) if lhs == rhs => {
                let r_a = self.get_register();
                self.push_store(r_a, m_y);
                self.out.push(op(r_x, r_a, r_a));
                self.release_reg(r_x);
                self.bind_register(lhs, r_a);
            }
            (Allocation::Memory(m_y), Allocation::Memory(m_z)) => {
                let r_a = self.get_register();
                let r_b = self.get_register();

                self.push_store(r_a, m_y);
                self.push_store(r_b, m_z);
                self.out.push(op(r_x, r_a, r_b));
                self.release_reg(r_x);
                self.bind_register(lhs, r_a);
                self.bind_register(rhs, r_b);
            }
            (Allocation::Unassigned, Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);
            }
            (Allocation::Register(r_y), Allocation::Unassigned) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);
            }
            (Allocation::Unassigned, Allocation::Unassigned) if lhs == rhs => {
                self.out.push(op(r_x, r_x, r_x));
                self.rebind_register(lhs, r_x);
            }
            (Allocation::Unassigned, Allocation::Unassigned) => {
                let r_a = self.get_register();

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                self.bind_register(rhs, r_a);
            }
            (Allocation::Unassigned, Allocation::Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);
                assert!(lhs != rhs);

                self.push_store(r_a, m_z);
                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                self.bind_register(rhs, r_a);
            }
            (Allocation::Memory(m_y), Allocation::Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);
                assert!(lhs != rhs);

                self.push_store(r_a, m_y);
                self.out.push(op(r_x, r_a, r_x));
                self.bind_register(lhs, r_a);
                self.rebind_register(rhs, r_x);
            }
        }
    }

    /// Lowers a function taking one register and one immediate into an
    /// [`RegOp`], pushing it to the internal tape.
    #[inline(always)]
    fn op_reg_imm(&mut self, op: SsaOp) {
        let (out, arg, imm, op): (_, _, _, fn(u8, u8, f32) -> RegOp) = match op
        {
            SsaOp::AddRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::AddRegImm)
            }
            SsaOp::SubRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::SubRegImm)
            }
            SsaOp::SubImmReg(out, arg, imm) => {
                (out, arg, imm, RegOp::SubImmReg)
            }
            SsaOp::MulRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::MulRegImm)
            }
            SsaOp::DivRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::DivRegImm)
            }
            SsaOp::DivImmReg(out, arg, imm) => {
                (out, arg, imm, RegOp::DivImmReg)
            }
            SsaOp::MinRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::MinRegImm)
            }
            SsaOp::MaxRegImm(out, arg, imm) => {
                (out, arg, imm, RegOp::MaxRegImm)
            }
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, |out, arg| op(out, arg, imm));
    }

    #[inline(always)]
    fn op_out_only(&mut self, out: u32, op: impl Fn(u8) -> RegOp) {
        let r_x = self.get_out_reg(out);
        self.out.push(op(r_x));
        self.release_reg(r_x);
    }

    /// Pushes a [`CopyImm`](crate::compiler::RegOp::CopyImm) operation to the
    /// tape
    #[inline(always)]
    fn op_copy_imm(&mut self, out: u32, imm: f32) {
        self.op_out_only(out, |out| RegOp::CopyImm(out, imm));
    }

    /// Pushes an [`Input`](crate::compiler::RegOp::Input) operation to the tape
    #[inline(always)]
    fn op_input(&mut self, out: u32, i: u8) {
        self.op_out_only(out, |out| RegOp::Input(out, i));
    }

    /// Pushes an [`Var`](crate::compiler::RegOp::Var) operation to the tape
    #[inline(always)]
    fn op_var(&mut self, out: u32, i: u32) {
        self.op_out_only(out, |out| RegOp::Var(out, i));
    }
}
