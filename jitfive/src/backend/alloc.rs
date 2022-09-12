use crate::backend::asm::AsmOp;
use crate::backend::tape::TapeOp;
use crate::util::lru::Lru;

use arrayvec::ArrayVec;

#[derive(Copy, Clone, Debug)]
enum Allocation {
    Register(u8),
    Memory(u32),
    Unassigned,
}

pub struct RegisterAllocator {
    /// Map from the index in the original (globally allocated) tape to a
    /// specific register or memory slot.
    allocations: Vec<u32>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `usize::MAX` if the register is currently unused.
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
    register_lru: Lru<{ u8::MAX as usize + 1 }>,

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

    /// Total allocated slots
    ///
    /// This will be <= the number of clauses in the tape, because we can often
    /// reuse slots.
    total_slots: u32,

    /// Output slots, assembled in reverse order
    out: Vec<AsmOp>,
}

impl RegisterAllocator {
    /// Builds a new `RegisterAllocator`.
    ///
    /// Upon construction, nothing is bound; calling `bind_initial_register` may
    /// be necessary.
    pub fn new(reg_limit: u8, size: usize) -> Self {
        Self {
            allocations: vec![u32::MAX; size],

            registers: [u32::MAX; u8::MAX as usize],
            register_lru: Lru::new(reg_limit as usize),

            reg_limit,
            spare_registers: ArrayVec::new(),
            spare_memory: Vec::with_capacity(1024),

            total_slots: 0,
            out: Vec::with_capacity(1024),
        }
    }

    /// Binds the SSA register 0 to local register 0
    pub fn bind_initial_register(&mut self) {
        self.bind_register(0, 0);
        self.total_slots += 1;
    }

    /// Claims the internal `Vec<AsmOp>`
    pub fn take(self) -> Vec<AsmOp> {
        self.out
    }

    /// Returns an available memory slot.
    ///
    /// We can *always* call this, because we treat memory as unlimited.
    ///
    /// > If there's one thing I love  
    /// > It's an infinite resource  
    /// > If there's one thing worth loving  
    /// > It's a surplus of supplies
    fn get_memory(&mut self) -> u32 {
        if let Some(p) = self.spare_memory.pop() {
            p
        } else {
            let out = self.total_slots;
            self.total_slots += 1;
            out
        }
    }

    /// Finds the oldest register
    ///
    /// This is useful when deciding which register to evict to make room
    fn oldest_reg(&mut self) -> u8 {
        self.register_lru.pop() as u8
    }

    /// Returns the slot allocated to the given node
    ///
    /// The input is an SSA assignment (i.e. an assignment in the global `Tape`)
    ///
    /// If the output is a register, then it's poked to update recency
    fn get_allocation(&mut self, n: u32) -> Allocation {
        match self.allocations[n as usize] {
            i if i < self.reg_limit as u32 => {
                self.register_lru.poke(i as usize);
                Allocation::Register(i as u8)
            }
            u32::MAX => Allocation::Unassigned,
            i => Allocation::Memory(i),
        }
    }

    /// Return an unoccupied register, if available
    fn get_spare_register(&mut self) -> Option<u8> {
        self.spare_registers.pop().or_else(|| {
            if self.total_slots < self.reg_limit as u32 {
                let reg = self.total_slots;
                assert!(self.registers[reg as usize] == u32::MAX);
                self.total_slots += 1;
                Some(reg.try_into().unwrap())
            } else {
                None
            }
        })
    }

    fn get_register(&mut self) -> u8 {
        if let Some(reg) = self.get_spare_register() {
            assert_eq!(self.registers[reg as usize], u32::MAX);
            self.register_lru.poke(reg as usize);
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

            self.out.push(AsmOp::Load(reg, mem, line!()));
            reg
        }
    }

    fn rebind_register(&mut self, n: u32, reg: u8) {
        assert!(self.allocations[n as usize] >= self.reg_limit as u32);
        assert!(self.registers[reg as usize] != u32::MAX);

        let prev_node = self.registers[reg as usize];
        self.allocations[prev_node as usize] = u32::MAX;

        // Bind the register and update its use time
        self.registers[reg as usize] = n;
        self.allocations[n as usize] = reg as u32;
        self.register_lru.poke(reg as usize);
    }

    fn bind_register(&mut self, n: u32, reg: u8) {
        assert!(self.allocations[n as usize] >= self.reg_limit as u32);
        assert!(self.registers[reg as usize] == u32::MAX);

        // Bind the register and update its use time
        self.registers[reg as usize] = n;
        self.allocations[n as usize] = reg as u32;
        self.register_lru.poke(reg as usize);
    }

    /// Release a register back to the pool of spares
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

    fn release_mem(&mut self, mem: u32) {
        assert!(mem >= self.reg_limit as u32);
        self.spare_memory.push(mem);
        // This leaves self.allocations[...] stil pointing to the memory slot,
        // but that's okay, because it should never be used
    }

    /// Lowers an operation that uses a single register into an `AsmOp`
    ///
    /// This is surprisingly tricky; see the source code for the table of 6
    /// possible situations.
    pub fn op_reg(&mut self, out: u32, arg: u32, op: TapeOp) {
        let op: fn(u8, u8) -> AsmOp = match op {
            TapeOp::NegReg => AsmOp::NegReg,
            TapeOp::AbsReg => AsmOp::AbsReg,
            TapeOp::RecipReg => AsmOp::RecipReg,
            TapeOp::SqrtReg => AsmOp::SqrtReg,
            TapeOp::SquareReg => AsmOp::SquareReg,
            TapeOp::CopyReg => AsmOp::CopyReg,
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, op);
    }

    fn op_reg_fn(&mut self, out: u32, arg: u32, op: impl Fn(u8, u8) -> AsmOp) {
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
        use Allocation::*;
        match (self.get_allocation(out), self.get_allocation(arg)) {
            (Register(r_x), Register(r_y)) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            (Register(r_x), Memory(m_y)) => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);

                self.out.push(AsmOp::Store(r_x, m_y, line!()));
                self.release_mem(m_y);
            }
            (Register(r_x), Unassigned) => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);
            }
            (Memory(m_x), Register(r_y)) => {
                let r_a = self.get_register();
                assert!(r_a != r_y);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_y));
                self.release_reg(r_a);
            }
            (Memory(m_x), Memory(m_y)) => {
                let r_a = self.get_register();

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a));
                self.rebind_register(arg, r_a);

                self.out.push(AsmOp::Store(r_a, m_y, line!()));
                self.release_mem(m_y);
            }
            (Memory(m_x), Unassigned) => {
                let r_a = self.get_register();

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a));
                self.rebind_register(arg, r_a);
            }
            (Unassigned, _) => panic!("Cannot have unassigned output"),
        }
    }

    /// Lowers a two-register operation into `AsmOp` and stores them
    ///
    /// Inputs are SSA registers from a `Tape`, i.e. globally addressed.
    ///
    /// This is trickier than it sounds, because it could require evicting
    /// previous registers to make room!  For all the gory details, look at the
    /// table in the source code, showing the 18 (!) possible configurations.
    pub fn op_reg_reg(&mut self, out: u32, lhs: u32, rhs: u32, op: TapeOp) {
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
        let op: fn(u8, u8, u8) -> AsmOp = match op {
            TapeOp::AddRegReg => AsmOp::AddRegReg,
            TapeOp::SubRegReg => AsmOp::SubRegReg,
            TapeOp::MulRegReg => AsmOp::MulRegReg,
            TapeOp::MinRegReg => AsmOp::MinRegReg,
            TapeOp::MaxRegReg => AsmOp::MaxRegReg,
            _ => panic!("Bad opcode: {op:?}"),
        };
        use Allocation::*;
        match (
            self.get_allocation(out),
            self.get_allocation(lhs),
            self.get_allocation(rhs),
        ) {
            (Register(r_x), Register(r_y), Register(r_z)) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (Register(r_x), Memory(m_y), Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);

                self.out.push(AsmOp::Store(r_x, m_y, line!()));
                self.release_mem(m_y);
            }
            (Register(r_x), Register(r_y), Memory(m_z)) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);

                self.out.push(AsmOp::Store(r_x, m_z, line!()));
                self.release_mem(m_z);
            }
            (Register(r_x), Memory(m_y), Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                self.bind_register(rhs, r_a);

                self.out.push(AsmOp::Store(r_x, m_y, line!()));
                self.release_mem(m_y);

                self.out.push(AsmOp::Store(r_a, m_z, line!()));
                self.release_mem(m_z);
            }
            (Register(r_x), Unassigned, Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);
            }
            (Register(r_x), Register(r_y), Unassigned) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);
            }
            (Register(r_x), Unassigned, Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                self.bind_register(rhs, r_a);
            }
            (Register(r_x), Unassigned, Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                self.bind_register(rhs, r_a);

                self.out.push(AsmOp::Store(r_a, m_z, line!()));
                self.release_mem(m_z);
            }
            (Register(r_x), Memory(m_y), Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_a, r_x));
                self.bind_register(lhs, r_a);
                self.rebind_register(rhs, r_x);

                self.out.push(AsmOp::Store(r_a, m_y, line!()));
                self.release_mem(m_y);
            }

            (Memory(m_x), Register(r_y), Register(r_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_y);
                assert!(r_a != r_z);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_y, r_z));
                self.release_reg(r_a);
            }
            (Memory(m_x), Register(r_y), Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_y);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_y, r_a));
                self.rebind_register(rhs, r_a);

                self.out.push(AsmOp::Store(r_a, m_z, line!()));
                self.release_mem(m_z);
            }
            (Memory(m_x), Memory(m_y), Register(r_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_z);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_z));
                self.rebind_register(lhs, r_a);

                self.out.push(AsmOp::Store(r_a, m_y, line!()));
                self.release_mem(m_y);
            }
            (Memory(m_x), Memory(m_y), Memory(m_z)) => {
                let r_a = self.get_register();
                let r_b = self.get_register();
                assert!(r_a != r_b);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_b));
                self.rebind_register(lhs, r_a);
                self.bind_register(rhs, r_b);

                self.out.push(AsmOp::Store(r_a, m_y, line!()));
                self.out.push(AsmOp::Store(r_b, m_z, line!()));
                self.release_mem(m_y);
                self.release_mem(m_z);
            }
            (Memory(m_x), Register(r_y), Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_y);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_y, r_a));
                self.rebind_register(rhs, r_a);
            }
            (Memory(m_x), Unassigned, Register(r_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_z);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_z));
                self.rebind_register(lhs, r_a);
            }
            (Memory(m_x), Unassigned, Unassigned) => {
                let r_a = self.get_register();
                let r_b = self.get_register();
                assert!(r_a != r_b);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_b));
                self.rebind_register(lhs, r_a);
                self.bind_register(rhs, r_b);
            }
            (Memory(m_x), Memory(m_y), Unassigned) => {
                let r_a = self.get_register();
                let r_b = self.get_register();
                assert!(r_a != r_b);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_b));
                self.rebind_register(lhs, r_a);
                self.bind_register(rhs, r_b);

                self.out.push(AsmOp::Store(r_a, m_y, line!()));
                self.release_mem(m_y);
            }
            (Memory(m_x), Unassigned, Memory(m_z)) => {
                let r_a = self.get_register();
                let r_b = self.get_register();
                assert!(r_a != r_b);

                self.out.push(AsmOp::Store(r_a, m_x, line!()));
                self.release_mem(m_x);
                self.bind_register(out, r_a);

                self.out.push(op(r_a, r_a, r_b));
                self.rebind_register(lhs, r_a);
                self.bind_register(rhs, r_b);

                self.out.push(AsmOp::Store(r_b, m_z, line!()));
                self.release_mem(m_z);
            }
            (Unassigned, _, _) => panic!("Cannot have unassigned output"),
        }
    }

    pub fn op_reg_imm(&mut self, out: u32, arg: u32, imm: f32, op: TapeOp) {
        let op: fn(u8, u8, f32) -> AsmOp = match op {
            TapeOp::AddRegImm => AsmOp::AddRegImm,
            TapeOp::SubRegImm => AsmOp::SubRegImm,
            TapeOp::SubImmReg => AsmOp::SubImmReg,
            TapeOp::MulRegImm => AsmOp::MulRegImm,
            TapeOp::MinRegImm => AsmOp::MinRegImm,
            TapeOp::MaxRegImm => AsmOp::MaxRegImm,
            _ => panic!("Bad opcode: {op:?}"),
        };
        self.op_reg_fn(out, arg, |out, arg| op(out, arg, imm));
    }

    fn op_out_only(&mut self, out: u32, op: impl Fn(u8) -> AsmOp) {
        use Allocation::*;
        match self.get_allocation(out) {
            Register(reg) => {
                self.out.push(op(reg));
            }
            Memory(mem) => {
                let r_a = self.get_register();

                self.out.push(AsmOp::Store(r_a, mem, line!()));
                self.release_mem(mem);
                self.bind_register(out, r_a);

                self.out.push(op(r_a));
                self.release_reg(r_a);
            }
            Unassigned => panic!("Cannot have unassigned output"),
        }
    }

    /// Lowers a `CopyImm` operation into the tape
    pub fn op_copy_imm(&mut self, out: u32, imm: f32) {
        self.op_out_only(out, |out| AsmOp::CopyImm(out, imm));
    }

    /// Lowers a `Input` operation into the tape
    pub fn op_input(&mut self, out: u32, i: u8) {
        self.op_out_only(out, |out| AsmOp::Input(out, i));
    }
}
