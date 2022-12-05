use crate::{
    ssa::Op as SsaOp,
    vm::{lru::Lru, Op, Tape},
};

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
    out: Tape,
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

            out: Tape::new(reg_limit),
        };
        out.bind_register(0, 0);
        out
    }

    pub fn empty() -> Self {
        Self {
            allocations: vec![],

            registers: [u32::MAX; u8::MAX as usize],
            register_lru: Lru::new(0),

            reg_limit: 0,
            spare_registers: ArrayVec::new(),
            spare_memory: vec![],

            out: Tape::new(0),
        }
    }

    /// Resets the internal state, reusing allocations if possible
    pub fn reset(&mut self, reg_limit: u8, size: usize) {
        self.allocations.fill(u32::MAX);
        self.allocations.resize(size, u32::MAX);
        self.registers.fill(u32::MAX);
        self.register_lru = Lru::new(reg_limit);
        self.reg_limit = reg_limit;
        self.spare_registers.clear();
        self.spare_memory.clear();
        self.out = Tape::new(reg_limit);
        self.bind_register(0, 0);
    }

    /// Resets internal state, reusing allocations and the provided tape
    pub fn reset_with_storage(
        &mut self,
        reg_limit: u8,
        size: usize,
        tape: Tape,
    ) {
        assert!(self.out.is_empty());
        self.reset(reg_limit, size);
        self.out = tape;
        self.out.reset(reg_limit);
    }

    /// Claims the internal `Vec<Op>`, leaving it empty
    #[inline]
    pub fn finalize(&mut self) -> Tape {
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
            assert!(out >= self.out.reg_limit.into());
            out
        }
    }

    /// Finds the oldest register
    ///
    /// This is useful when deciding which register to evict to make room
    #[inline]
    fn oldest_reg(&mut self) -> u8 {
        self.register_lru.pop() as u8
    }

    /// Returns the slot allocated to the given node
    ///
    /// The input is an SSA assignment (i.e. an assignment in the global `Tape`)
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

            self.out.push(Op::Load(reg, mem));
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
    /// [`Op`](crate::asm::Op), pushing it to the internal tape.
    ///
    /// This may also push `Load` or `Store` instructions to the internal tape,
    /// if there aren't enough spare registers.
    pub fn op_reg(&mut self, out: u32, arg: u32, op: SsaOp) {
        let op: fn(u8, u8) -> Op = match op {
            SsaOp::NegReg => Op::NegReg,
            SsaOp::AbsReg => Op::AbsReg,
            SsaOp::RecipReg => Op::RecipReg,
            SsaOp::SqrtReg => Op::SqrtReg,
            SsaOp::SquareReg => Op::SquareReg,
            SsaOp::CopyReg => Op::CopyReg,
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, op);
    }

    fn push_store(&mut self, reg: u8, mem: u32) {
        self.out.push(Op::Store(reg, mem));
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

    #[inline]
    fn op_reg_fn(&mut self, out: u32, arg: u32, op: impl Fn(u8, u8) -> Op) {
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
        //   r_x | m_y | store r_x -> m_y
        //       |     | r_x = op r_x
        //       |     |
        //       |     | Afterward, r_x points to the former m_y
        //  -----|-----|----------------------------------------------------
        //   r_x |  U  | r_x = op r_x
        //       |     |
        //       |     | Afterward, r_x points to the arg
        //  -----|-----|----------------------------------------------------
        //   m_x | r_y | r_a = op r_y
        //       |     | store r_a -> m_x
        //       |     | [load r_a <- m_a]
        //       |     |
        //       |     | Afterward, r_a and m_x are free, [m_a points to the
        //       |     | former r_a]
        //  -----|-----|----------------------------------------------------
        //   m_x | m_y | store r_a -> m_y
        //       |     | r_a = op rA
        //       |     | store r_a -> m_x
        //       |     | [load r_a <- m_a]
        //       |     |
        //       |     | Afterwards, r_a points to arg, m_x and m_y are free,
        //       |     | [and m_a points to the former r_a]
        //  -----|-----|----------------------------------------------------
        //   m_x |  U  | r_a = op rA
        //       |     | store r_a -> m_x
        //       |     | [load r_a <- m_a]
        //       |     |
        //       |     | Afterwards, r_a points to the arg, m_x is free,
        //       |     | [and m_a points to the former r_a]
        //  -----|-----|----------------------------------------------------
        let r_x = self.get_out_reg(out);
        match self.get_allocation(arg) {
            Allocation::Register(r_y) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            Allocation::Memory(m_y) => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);

                self.push_store(r_x, m_y);
            }
            Allocation::Unassigned => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);
            }
        }
    }

    /// Lowers a two-register operation into an [`Op`](crate::asm::Op),
    /// pushing it to the internal tape.
    ///
    /// Inputs are SSA registers from a [`Tape`](crate::tape::Tape), i.e.
    /// globally addressed.
    ///
    /// If there aren't enough spare registers, this may also push `Load` or
    /// `Store` instructions to the internal tape.  It's trickier than it
    /// sounds; look at the source code for a table showing all 18 (!) possible
    /// configurations.
    #[inline]
    pub fn op_reg_reg(&mut self, out: u32, lhs: u32, rhs: u32, op: SsaOp) {
        // Looking at this horrific table, you may be tempted to think "surely
        // there's a clean abstraction that wraps this up in a few functions".
        // You may be right, but I spent a few days chasing down terrible memory
        // load/store ordering bugs, and decided that the brute-force approach
        // was the right one.
        //
        //   out | lhs | rhs | what do?
        //  ================================================================
        //  r_x  | r_y  | r_z  | r_x = op r_y r_z
        //       |      |      |
        //       |      |      | Afterwards, r_x is free
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | r_z  | store r_x -> m_y
        //       |      |      | r_x = op r_x r_z
        //       |      |      |
        //       |      |      | Afterwards, r_x points to the former m_y, and
        //       |      |      | m_y is free
        //  -----|------|------|----------------------------------------------
        //  r_x  | r_y  | m_z  | ibid
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | m_z  | store r_x -> m_y
        //       |      |      | store r_a -> m_z
        //       |      |      | r_x = op r_x r_a
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
        //  r_x  | U    | m_z  | store r_a -> m_z
        //       |      |      | r_x = op r_x r_a
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterward, r_x points to the lhs, r_a points to
        //       |      |      | rhs, m_z is free, [and m_a points to the former
        //       |      |      | r_a]
        //  -----|------|------|----------------------------------------------
        //  r_x  | m_y  | U    | ibid
        //  =====|======|======|==============================================
        //   m_x | r_y  | r_z  | r_a = op r_y r_z
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_a and m_x are free, [m_a points
        //       |      |      | to the former r_a}
        //  -----|------|------|----------------------------------------------
        //   m_x | r_y  | m_z  | store r_a -> m_z
        //       |      |      | r_a = op r_y rA
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_a points to rhs, m_z and m_x are
        //       |      |      | free, [and m_a points to the former r_a]
        //  -----|------|------|----------------------------------------------
        //   m_x | m_y  | r_z  | ibid
        //  -----|------|------|----------------------------------------------
        //   m_x | m_y  | m_z  | store r_a -> m_y
        //       |      |      | store r_b -> m_z
        //       |      |      | r_a = op rA r_b
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      | [load r_b <- m_b]
        //       |      |      |
        //       |      |      | Afterwards, r_a points to lhs, r_b points to
        //       |      |      | rhs, m_x, m_y, m_z are all free, [m_a points to
        //       |      |      | the former r_a], [m_b points to the former r_b]
        //  -----|------|------|----------------------------------------------
        //   m_x | r_y  | U    | r_a = op r_y rA
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      |
        //       |      |      | Afterwards, r_a points to rhs, m_x is free,
        //       |      |      | [m_a points to the former r_a]
        //  -----|------|------|----------------------------------------------
        //   m_x |  U   | r_z  | ibid
        //  -----|------|------|----------------------------------------------
        //   m_x |  U   | U    | r_a = op rA r_b
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      | [load r_b <- m_b]
        //       |      |      |
        //       |      |      | Afterwards, r_a points to lhs, r_b points to
        //       |      |      | rhs, m_x is free, [m_a points to the former
        //       |      |      | r_a], [m_b points to the former r_b]
        //  -----|------|------|----------------------------------------------
        //   m_x | m_y  | U    | store r_a -> m_y
        //       |      |      | r_a = op rA r_b
        //       |      |      | store r_a -> m_x
        //       |      |      | [load r_a <- m_a]
        //       |      |      | [load r_b <- m_b]
        //       |      |      |
        //       |      |      | Afterwards, r_a points to lhs, r_b points to
        //       |      |      | rhs,  m_x and m_y are free, [m_a points to the
        //       |      |      | former r_a], [m_b points to the former r_b]
        //  -----|------|------|----------------------------------------------
        //   m_x  | U   | m_z  | ibid
        let op: fn(u8, u8, u8) -> Op = match op {
            SsaOp::AddRegReg => Op::AddRegReg,
            SsaOp::SubRegReg => Op::SubRegReg,
            SsaOp::MulRegReg => Op::MulRegReg,
            SsaOp::DivRegReg => Op::DivRegReg,
            SsaOp::MinRegReg => Op::MinRegReg,
            SsaOp::MaxRegReg => Op::MaxRegReg,
            _ => panic!("Bad opcode: {op:?}"),
        };
        let r_x = self.get_out_reg(out);
        match (self.get_allocation(lhs), self.get_allocation(rhs)) {
            (Allocation::Register(r_y), Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (Allocation::Memory(m_y), Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);

                self.push_store(r_x, m_y);
            }
            (Allocation::Register(r_y), Allocation::Memory(m_z)) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);

                self.push_store(r_x, m_z);
            }
            (Allocation::Memory(m_y), Allocation::Memory(m_z)) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }

                self.push_store(r_x, m_y);

                if lhs != rhs {
                    self.push_store(r_a, m_z);
                }
            }
            (Allocation::Unassigned, Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);
            }
            (Allocation::Register(r_y), Allocation::Unassigned) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);
            }
            (Allocation::Unassigned, Allocation::Unassigned) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }
            }
            (Allocation::Unassigned, Allocation::Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }

                self.push_store(r_a, m_z);
            }
            (Allocation::Memory(m_y), Allocation::Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_a, r_x));
                self.bind_register(lhs, r_a);
                if lhs != rhs {
                    self.rebind_register(rhs, r_x);
                }

                self.push_store(r_a, m_y);
            }
        }
    }

    /// Lowers a function taking one register and one immediate into an
    /// [`Op`](crate::asm::Op), pushing it to the internal tape.
    #[inline]
    pub fn op_reg_imm(&mut self, out: u32, arg: u32, imm: f32, op: SsaOp) {
        let op: fn(u8, u8, f32) -> Op = match op {
            SsaOp::AddRegImm => Op::AddRegImm,
            SsaOp::SubRegImm => Op::SubRegImm,
            SsaOp::SubImmReg => Op::SubImmReg,
            SsaOp::MulRegImm => Op::MulRegImm,
            SsaOp::DivRegImm => Op::DivRegImm,
            SsaOp::DivImmReg => Op::DivImmReg,
            SsaOp::MinRegImm => Op::MinRegImm,
            SsaOp::MaxRegImm => Op::MaxRegImm,
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, |out, arg| op(out, arg, imm));
    }

    #[inline]
    fn op_out_only(&mut self, out: u32, op: impl Fn(u8) -> Op) {
        let r_x = self.get_out_reg(out);
        self.out.push(op(r_x));
        self.release_reg(r_x);
    }

    /// Pushes a [`CopyImm`](crate::asm::Op::CopyImm) operation to the tape
    #[inline]
    pub fn op_copy_imm(&mut self, out: u32, imm: f32) {
        self.op_out_only(out, |out| Op::CopyImm(out, imm));
    }

    /// Pushes an [`Input`](crate::asm::Op::Input) operation to the tape
    #[inline]
    pub fn op_input(&mut self, out: u32, i: u8) {
        self.op_out_only(out, |out| Op::Input(out, i));
    }

    /// Pushes an [`Var`](crate::asm::Op::Var) operation to the tape
    #[inline]
    pub fn op_var(&mut self, out: u32, i: u32) {
        self.op_out_only(out, |out| Op::Var(out, i));
    }
}
