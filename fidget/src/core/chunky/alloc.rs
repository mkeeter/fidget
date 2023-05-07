use crate::{
    context::{self, Context, Node, VarNode},
    vm::{lru::Lru, Op, Tape},
};
use std::collections::BTreeMap;

use arrayvec::ArrayVec;

#[derive(Copy, Clone, Debug)]
enum AbstractAllocation {
    Register(u8),
    Memory(u32),
    Both(u8, u32),
}

#[derive(Copy, Clone, Debug)]
enum Allocation {
    Register(u8),
    Memory(u32),
}

/// Cheap and cheerful single-pass register allocation
pub struct RegisterAllocator<'a> {
    ctx: &'a Context,

    vars: BTreeMap<VarNode, u32>,
    var_names: BTreeMap<String, u32>,

    /// Map from the index in the original (globally allocated) tape to a
    /// specific register or memory slot.
    allocations: BTreeMap<Node, AbstractAllocation>,

    /// Map from a particular register to the original node that's using that
    /// register.  This is the reverse of `allocations`.
    registers: BTreeMap<u8, Node>,

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

impl<'a> RegisterAllocator<'a> {
    pub fn new(ctx: &'a Context, reg_limit: u8) -> Self {
        Self {
            ctx,

            vars: BTreeMap::new(),
            var_names: BTreeMap::new(),

            allocations: BTreeMap::new(),
            registers: BTreeMap::new(),

            register_lru: Lru::new(reg_limit),

            reg_limit,
            spare_registers: ArrayVec::new(),
            spare_memory: Vec::new(),

            out: Tape::new(reg_limit),
        }
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
            assert!(out >= self.out.reg_limit().into());
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
    /// If the output is a register, then it's poked to update recency
    #[inline]
    fn get_allocation(&mut self, n: Node) -> Option<Allocation> {
        match self.allocations.get(&n)? {
            AbstractAllocation::Register(reg)
            | AbstractAllocation::Both(reg, ..) => {
                self.register_lru.poke(*reg);
                Some(Allocation::Register(*reg))
            }
            AbstractAllocation::Memory(mem) => Some(Allocation::Memory(*mem)),
        }
    }

    /// Return an unoccupied register, if available
    #[inline]
    fn get_spare_register(&mut self) -> Option<u8> {
        self.spare_registers.pop().or_else(|| {
            if self.out.slot_count < self.reg_limit as u32 {
                let reg = self.out.slot_count.try_into().unwrap();
                assert!(!self.registers.contains_key(&reg));
                self.out.slot_count += 1;
                Some(reg.try_into().unwrap())
            } else {
                None
            }
        })
    }

    /// Returns an unbound (available) register
    ///
    /// This may evict a currently-bound register, in which case a `Load`
    /// operation is pushed to the tape.
    #[inline]
    fn get_register(&mut self) -> u8 {
        if let Some(reg) = self.get_spare_register() {
            assert!(!self.registers.contains_key(&reg));
            self.register_lru.poke(reg);
            reg
        } else {
            // Slot is in memory, and no spare register is available
            let reg = self.oldest_reg();

            // Whoever was previously using you is in for a surprise
            let prev_node = self.registers[&reg];

            // If the previous user of this register _also_ has a memory slot
            // associated with it, then we'll reuse that slot here.
            let mem = match self.allocations.get(&prev_node) {
                Some(AbstractAllocation::Both(prev_reg, mem)) => {
                    assert_eq!(*prev_reg, reg);
                    *mem
                }
                Some(AbstractAllocation::Register(prev_reg)) => {
                    // Otherwise, assign a new slot for it.
                    assert_eq!(*prev_reg, reg);
                    self.get_memory()
                }
                v => panic!("invalid allocation {v:?}"),
            };

            // The previous node is now only bound to memory.
            self.allocations
                .insert(prev_node, AbstractAllocation::Memory(mem));

            // The output register is now unassigned
            self.registers.remove(&reg);

            // This operation keeps things in sync
            self.out.push(Op::Load(reg, mem));
            reg
        }
    }

    /// Remove the register associated with the given node
    fn alloc_remove_reg(&mut self, n: Node) -> u8 {
        match self.allocations.get_mut(&n).cloned() {
            Some(AbstractAllocation::Both(reg, mem)) => {
                self.allocations.insert(n, AbstractAllocation::Memory(mem));
                reg
            }
            Some(AbstractAllocation::Register(reg)) => {
                self.allocations.remove(&n);
                reg
            }
            v => panic!("cannot remove register from {v:?}"),
        }
    }

    /// Moves the given node's binding from memory to a register
    #[inline]
    fn rebind_register(&mut self, n: Node, reg: u8) {
        assert!(self.registers.contains_key(&reg));

        let prev_node = self.registers[&reg];
        let prev_reg = self.alloc_remove_reg(prev_node);
        assert_eq!(reg, prev_reg);
        self.registers.remove(&reg);

        self.bind_register(n, reg);
    }

    #[inline]
    fn bind_register(&mut self, n: Node, reg: u8) {
        assert!(!self.registers.contains_key(&reg));

        // Bind the register and update its use time
        self.registers.insert(reg, n);
        self.register_lru.poke(reg);
        match self.allocations.get_mut(&n).cloned() {
            Some(AbstractAllocation::Memory(mem)) => {
                self.allocations
                    .insert(n, AbstractAllocation::Both(reg, mem));
            }
            None => {
                self.allocations
                    .insert(n, AbstractAllocation::Register(reg));
            }
            v => panic!("cannot insert register into {v:?}"),
        }
    }

    /// Release a register back to the pool of spares
    #[inline]
    fn release_reg(&mut self, reg: u8) {
        // Release the output register, so it could be used for inputs
        assert!(reg < self.reg_limit);

        let node = self.registers.remove(&reg).unwrap();
        self.spare_registers.push(reg);

        // Modifying self.allocations isn't strictly necessary, but could help
        // us detect logical errors (since it should never be used after this)
        self.allocations.remove(&node);
    }

    /// Lowers an operation into an [`Op`](crate::vm::Op), pushing it to the
    /// internal tape.
    ///
    /// This may also push `Load` or `Store` instructions to the internal tape,
    /// if there aren't enough spare registers.
    #[inline(always)]
    pub fn op(&mut self, node: Node) {
        match *self.ctx.get_op(node).unwrap() {
            context::Op::Input(v) => {
                let arg = match self.ctx.get_var_by_index(v).unwrap() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    i => panic!("Unexpected input index: {i}"),
                };
                self.op_input(node, arg);
            }
            context::Op::Var(v) => {
                use std::collections::btree_map::Entry;
                let next_var = self.vars.len().try_into().unwrap();
                let arg = match self.vars.entry(v) {
                    Entry::Vacant(e) => {
                        e.insert(next_var);
                        let name =
                            self.ctx.get_var_by_index(v).unwrap().to_owned();
                        self.var_names.insert(name, next_var);
                        next_var
                    }
                    Entry::Occupied(a) => *a.get(),
                };
                self.op_var(node, arg);
            }

            context::Op::Const(..) => {
                // Do nothing; constants are always inlined
            }

            context::Op::Unary(op, arg) => {
                if self.ctx.const_value(arg).unwrap().is_some() {
                    panic!("cannot handle f(imm)");
                }
                let op = match op {
                    context::UnaryOpcode::Neg => Op::NegReg,
                    context::UnaryOpcode::Abs => Op::AbsReg,
                    context::UnaryOpcode::Recip => Op::RecipReg,
                    context::UnaryOpcode::Sqrt => Op::SqrtReg,
                    context::UnaryOpcode::Square => Op::SquareReg,
                };
                self.op_reg_fn(node, arg, op);
            }

            context::Op::Binary(op, lhs, rhs) => {
                type RegFn = fn(u8, u8, u8) -> Op;
                type ImmFn = fn(u8, u8, f32) -> Op;
                let f: (RegFn, ImmFn, ImmFn) = match op {
                    context::BinaryOpcode::Add => {
                        (Op::AddRegReg, Op::AddRegImm, Op::AddRegImm)
                    }
                    context::BinaryOpcode::Sub => {
                        (Op::SubRegReg, Op::SubRegImm, Op::SubImmReg)
                    }
                    context::BinaryOpcode::Mul => {
                        (Op::MulRegReg, Op::MulRegImm, Op::MulRegImm)
                    }
                    context::BinaryOpcode::Div => {
                        (Op::DivRegReg, Op::DivRegImm, Op::DivImmReg)
                    }
                    context::BinaryOpcode::Min => {
                        (Op::MinRegReg, Op::MinRegImm, Op::MinRegImm)
                    }
                    context::BinaryOpcode::Max => {
                        (Op::MaxRegReg, Op::MaxRegImm, Op::MaxRegImm)
                    }
                };
                match (
                    self.ctx.const_value(lhs).unwrap(),
                    self.ctx.const_value(rhs).unwrap(),
                ) {
                    (None, None) => self.op_reg_reg_fn(node, lhs, rhs, f.0),
                    (None, Some(rhs)) => {
                        self.op_reg_fn(node, lhs, |a, b| {
                            (f.2)(a, b, rhs as f32)
                        });
                    }
                    (Some(lhs), None) => {
                        self.op_reg_fn(node, rhs, |a, b| {
                            (f.2)(a, b, lhs as f32)
                        });
                    }
                    (Some(_lhs), Some(_rhs)) => {
                        panic!("Cannot handle f(imm, imm)");
                    }
                }
            }
        }
    }

    fn push_store(&mut self, reg: u8, mem: u32) {
        assert!(mem >= self.reg_limit as u32);
        self.out.push(Op::Store(reg, mem));
        self.spare_memory.push(mem);
    }

    /// Returns a register that is bound to the given SSA input
    ///
    /// If the given SSA input is not already bound to a register, then we
    /// evict the oldest register using `Self::get_register`, with the
    /// appropriate set of LOAD/STORE operations.
    #[inline]
    fn get_out_reg(&mut self, out: Node) -> u8 {
        match *self
            .allocations
            .get(&out)
            .expect("out register must be bound")
        {
            AbstractAllocation::Register(r_x) => {
                assert_eq!(self.registers.get(&r_x), Some(&out));
                r_x
            }
            AbstractAllocation::Memory(m_x) => {
                let r_a = self.get_register();

                self.push_store(r_a, m_x);
                self.bind_register(out, r_a);
                r_a
            }
            AbstractAllocation::Both(r_x, m_x) => {
                self.push_store(r_x, m_x);
                assert_eq!(self.registers.get(&r_x), Some(&out));
                r_x
            }
        }
    }

    #[inline(always)]
    fn op_reg_fn(&mut self, out: Node, arg: Node, op: impl Fn(u8, u8) -> Op) {
        let r_x = self.get_out_reg(out);
        match self.get_allocation(arg) {
            Some(Allocation::Register(r_y)) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            Some(Allocation::Memory(..)) | None => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);
            }
        }
    }

    /// Lowers a two-register operation into an [`Op`](crate::vm::Op),
    /// pushing it to the internal tape.
    ///
    /// Inputs are SSA registers from a [`Tape`], i.e. globally addressed.
    ///
    /// If there aren't enough spare registers, this may also push `Load` or
    /// `Store` instructions to the internal tape.  It's trickier than it
    /// sounds; look at the source code for a table showing all 18 (!) possible
    /// configurations.
    #[inline(always)]
    fn op_reg_reg_fn(
        &mut self,
        out: Node,
        lhs: Node,
        rhs: Node,
        op: impl Fn(u8, u8, u8) -> Op,
    ) {
        let r_x = self.get_out_reg(out);
        match (self.get_allocation(lhs), self.get_allocation(rhs)) {
            (
                Some(Allocation::Register(r_y)),
                Some(Allocation::Register(r_z)),
            ) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (
                Some(Allocation::Memory(..)) | None,
                Some(Allocation::Register(r_z)),
            ) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);
            }
            (
                Some(Allocation::Register(r_y)),
                Some(Allocation::Memory(..)) | None,
            ) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);
            }
            (
                Some(Allocation::Memory(..)) | None,
                Some(Allocation::Memory(..)) | None,
            ) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }
            }
        }
    }

    #[inline(always)]
    fn op_out_only(&mut self, out: Node, op: impl Fn(u8) -> Op) {
        let r_x = self.get_out_reg(out);
        self.out.push(op(r_x));
        self.release_reg(r_x);
    }

    /// Pushes a [`CopyImm`](crate::vm::Op::CopyImm) operation to the tape
    #[inline(always)]
    fn op_copy_imm(&mut self, out: Node, imm: f32) {
        self.op_out_only(out, |out| Op::CopyImm(out, imm));
    }

    /// Pushes an [`Input`](crate::vm::Op::Input) operation to the tape
    #[inline(always)]
    fn op_input(&mut self, out: Node, i: u8) {
        self.op_out_only(out, |out| Op::Input(out, i));
    }

    /// Pushes an [`Var`](crate::vm::Op::Var) operation to the tape
    #[inline(always)]
    fn op_var(&mut self, out: Node, i: u32) {
        self.op_out_only(out, |out| Op::Var(out, i));
    }

    #[inline]
    pub fn tape_len(&self) -> usize {
        self.out.len()
    }
}
