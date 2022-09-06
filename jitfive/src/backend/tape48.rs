use std::collections::{BTreeMap, BTreeSet};

use crate::scheduled::Scheduled;
use crate::{
    backend::common::{Choice, NodeIndex, Op, VarIndex},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    util::indexed::IndexMap,
};

#[derive(Copy, Clone, Debug)]
pub enum ClauseOp48 {
    /// Reads one of the inputs (X, Y, Z)
    Input(u8),

    NegReg(u32),
    AbsReg(u32),
    RecipReg(u32),
    SqrtReg(u32),
    SquareReg(u32),

    /// Copies the given register
    ///
    /// (this is only useful in an `AllocatedTape`)
    CopyReg(u32),

    /// Add a register and an immediate
    AddRegImm(u32, f32),
    /// Multiply a register and an immediate
    MulRegImm(u32, f32),
    /// Subtract a register from an immediate
    SubImmReg(u32, f32),
    /// Subtract an immediate from a register
    SubRegImm(u32, f32),
    /// Compute the minimum of a register and an immediate
    MinRegImm(u32, f32),
    /// Compute the maximum of a register and an immediate
    MaxRegImm(u32, f32),

    AddRegReg(u32, u32),
    MulRegReg(u32, u32),
    SubRegReg(u32, u32),
    MinRegReg(u32, u32),
    MaxRegReg(u32, u32),

    /// Copy an immediate to a register
    CopyImm(f32),
}

/// Tape storing 48-bit (12-byte) operations:
/// - 4-byte opcode
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs are implicitly stored based on clause index in `tape`, e.g. the
/// first item in the tape writes to slot 0.
#[derive(Clone, Debug)]
pub struct Tape {
    /// Raw instruction tape, stored in reverse evaluation order
    pub tape: Vec<ClauseOp48>,

    /// `last_used[i]` is the last use of slot `i` during forward evaluation
    last_used: Vec<usize>,

    /// The number of nodes which store values in the choice array during
    /// interval evaluation.
    pub choice_count: usize,
}

impl Tape {
    /// Builds an evaluator which takes a (read-only) reference to this tape
    pub fn get_evaluator(&self) -> TapeEval {
        TapeEval {
            tape: self,
            slots: vec![0.0; self.tape.len()],
        }
    }

    pub fn alloc(&self, reg_limit: usize) -> AllocatedTape {
        let mut t = TapeAllocator::new(self, reg_limit);
        let mut tape = vec![];
        for i in &mut t {
            tape.push(i);
        }
        AllocatedTape {
            tape,
            choice_count: self.choice_count,
            total_slots: t.total_slots,
            out_slot: t.allocations[self.tape.len() - 1],
        }
    }

    /// Build a new tape from a pre-scheduled set of instructions
    pub fn new(t: &Scheduled) -> Self {
        Self::from_builder(TapeBuilder::new(t))
    }

    fn from_builder(mut builder: TapeBuilder) -> Self {
        let mut out = builder.run();
        out.reverse();
        Self {
            tape: out,
            choice_count: builder.choice_count,
            last_used: builder.last_used,
        }
    }

    pub fn pretty_print(&self) {
        for (i, op) in self.tape.iter().rev().enumerate() {
            print!("${} = ", i);
            use ClauseOp48::*;
            match op {
                Input(i) => println!("%{}", i),
                NegReg(arg) => println!("NEG ${}", arg),
                AbsReg(arg) => println!("ABS ${}", arg),
                RecipReg(arg) => println!("RECIP ${}", arg),
                SqrtReg(arg) => println!("SQRT ${}", arg),
                CopyReg(arg) => println!("COPY ${}", arg),
                SquareReg(arg) => println!("SQUARE ${}", arg),
                AddRegReg(lhs, rhs) => println!("ADD ${} ${}", lhs, rhs),
                MulRegReg(lhs, rhs) => println!("MUL ${} ${}", lhs, rhs),
                SubRegReg(lhs, rhs) => println!("SUB ${} ${}", lhs, rhs),
                MinRegReg(lhs, rhs) => println!("MIN ${} ${}", lhs, rhs),
                MaxRegReg(lhs, rhs) => println!("MAX ${} ${}", lhs, rhs),
                AddRegImm(arg, imm) => println!("ADD ${} {}", arg, imm),
                MulRegImm(arg, imm) => println!("MUL ${} {}", arg, imm),
                SubImmReg(arg, imm) => println!("SUB {} ${}", imm, arg),
                SubRegImm(arg, imm) => println!("SUB ${} {}", arg, imm),
                MinRegImm(arg, imm) => println!("MIN ${} {}", arg, imm),
                MaxRegImm(arg, imm) => println!("MAX ${} {}", arg, imm),
                CopyImm(imm) => println!("{}", imm),
            }
        }
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        let mut active = vec![false; self.tape.len()];
        let mut choice_iter = choices.iter().rev();
        active[self.tape.len() - 1] = true;

        // Reverse pass to track activity
        let i = (0..self.tape.len()).rev();
        for (index, op) in i.zip(self.tape.iter().cloned()) {
            use ClauseOp48::*;
            if !active[index] {
                if matches!(
                    op,
                    MinRegReg(..)
                        | MaxRegReg(..)
                        | MinRegImm(..)
                        | MaxRegImm(..)
                ) {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            match op {
                Input(..) | CopyImm(..) => (),
                AddRegReg(lhs, rhs)
                | MulRegReg(lhs, rhs)
                | SubRegReg(lhs, rhs) => {
                    active[lhs as usize] = true;
                    active[rhs as usize] = true;
                }

                NegReg(arg)
                | CopyReg(arg)
                | AbsReg(arg)
                | RecipReg(arg)
                | SqrtReg(arg)
                | SquareReg(arg)
                | AddRegImm(arg, ..)
                | MulRegImm(arg, ..)
                | SubImmReg(arg, ..)
                | SubRegImm(arg, ..) => {
                    active[arg as usize] = true;
                }
                MinRegImm(arg, ..) | MaxRegImm(arg, ..) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            active[arg as usize] = true;
                        }
                        Choice::Right => {
                            // Nothing to do here (will become CopyImm)
                        }
                        Choice::Both => {
                            active[arg as usize] = true;
                        }
                    }
                }

                MinRegReg(lhs, rhs) | MaxRegReg(lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            active[lhs as usize] = true;
                        }
                        Choice::Right => {
                            active[rhs as usize] = true;
                        }
                        Choice::Both => {
                            active[lhs as usize] = true;
                            active[rhs as usize] = true;
                        }
                    }
                }
            }
        }

        // Forward pass to build new tape
        let tape_iter = self.tape.iter().rev().cloned();
        let choice_iter = choices.iter();
        let active_iter = active.iter();
        let mut simplify = TapeSimplify {
            choice_count: 0,
            tape_iter,
            choice_iter,
            active_iter,
            remap: vec![],
            last_used: vec![],
        };

        let out = (&mut simplify).collect();

        Self {
            tape: out,
            choice_count: simplify.choice_count,
            last_used: simplify.last_used,
        }
    }
}

struct TapeSimplify<'a, I> {
    choice_count: usize,
    tape_iter: I,
    choice_iter: std::slice::Iter<'a, Choice>,
    active_iter: std::slice::Iter<'a, bool>,
    remap: Vec<u32>,
    last_used: Vec<usize>,
}

impl<'a, I> TapeSimplify<'a, I>
where
    I: Iterator<Item = ClauseOp48>,
{
    fn get(&mut self, i: u32) -> u32 {
        self.last_used[i as usize] = self.last_used.len();
        self.remap[i as usize]
    }

    fn step(&mut self, op: ClauseOp48) -> Option<ClauseOp48> {
        use ClauseOp48::*;

        let active = self.active_iter.next().unwrap();
        if !active {
            if matches!(
                op,
                MinRegReg(..) | MaxRegReg(..) | MinRegImm(..) | MaxRegImm(..)
            ) {
                self.choice_iter.next().unwrap();
            }
            self.remap.push(u32::MAX);
            return None;
        }

        let index = self.remap.len();
        let op = match op {
            Input(..) | CopyImm(..) => op,
            AddRegReg(lhs, rhs) => AddRegReg(self.get(lhs), self.get(rhs)),
            MulRegReg(lhs, rhs) => MulRegReg(self.get(lhs), self.get(rhs)),
            SubRegReg(lhs, rhs) => SubRegReg(self.get(lhs), self.get(rhs)),
            NegReg(arg) => NegReg(self.get(arg)),
            CopyReg(arg) => CopyReg(self.get(arg)),
            AbsReg(arg) => AbsReg(self.get(arg)),
            RecipReg(arg) => RecipReg(self.get(arg)),
            SqrtReg(arg) => SqrtReg(self.get(arg)),
            SquareReg(arg) => SquareReg(self.get(arg)),

            AddRegImm(arg, imm) => AddRegImm(self.get(arg), imm),
            MulRegImm(arg, imm) => MulRegImm(self.get(arg), imm),
            SubImmReg(arg, imm) => SubImmReg(self.get(arg), imm),
            SubRegImm(arg, imm) => SubRegImm(self.get(arg), imm),

            MinRegImm(arg, imm) | MaxRegImm(arg, imm) => {
                match self.choice_iter.next().unwrap() {
                    Choice::Left => {
                        self.remap.push(self.remap[arg as usize]);
                        return None;
                    }
                    Choice::Right => CopyImm(imm),
                    Choice::Both => {
                        self.choice_count += 1;
                        match op {
                            MinRegImm(arg, imm) => {
                                MinRegImm(self.get(arg), imm)
                            }
                            MaxRegImm(arg, imm) => {
                                MaxRegImm(self.get(arg), imm)
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            MinRegReg(lhs, rhs) | MaxRegReg(lhs, rhs) => {
                match self.choice_iter.next().unwrap() {
                    Choice::Left => {
                        self.remap.push(self.remap[lhs as usize]);
                        return None;
                    }
                    Choice::Right => {
                        self.remap.push(self.remap[rhs as usize]);
                        return None;
                    }
                    Choice::Both => {
                        self.choice_count += 1;
                        match op {
                            MinRegReg(lhs, rhs) => {
                                MinRegReg(self.get(lhs), self.get(rhs))
                            }
                            MaxRegReg(lhs, rhs) => {
                                MaxRegReg(self.get(lhs), self.get(rhs))
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        };
        self.last_used.push(usize::MAX);
        self.remap.push(index as u32);
        Some(op)
    }
}

impl<'a, I> Iterator for TapeSimplify<'a, I>
where
    I: Iterator<Item = ClauseOp48>,
{
    type Item = ClauseOp48;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let op = self.tape_iter.next()?;
            let r = self.step(op);
            if r.is_some() {
                break r;
            }
            // TODO: handle immediate-only tree?
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Workspace to evaluate a tape
pub struct TapeEval<'a> {
    tape: &'a Tape,
    slots: Vec<f32>,
}

impl<'a> TapeEval<'a> {
    fn v(&self, i: u32) -> f32 {
        self.slots[i as usize]
    }
    pub fn f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        for (i, &op) in self.tape.tape.iter().rev().enumerate() {
            self.slots[i] = match op {
                ClauseOp48::Input(i) => match i {
                    0 => x,
                    1 => y,
                    2 => z,
                    _ => panic!(),
                },
                ClauseOp48::NegReg(i) => -self.v(i),
                ClauseOp48::AbsReg(i) => self.v(i).abs(),
                ClauseOp48::RecipReg(i) => 1.0 / self.v(i),
                ClauseOp48::SqrtReg(i) => self.v(i).sqrt(),
                ClauseOp48::SquareReg(i) => self.v(i) * self.v(i),
                ClauseOp48::CopyReg(i) => self.v(i),

                ClauseOp48::AddRegReg(a, b) => self.v(a) + self.v(b),
                ClauseOp48::MulRegReg(a, b) => self.v(a) * self.v(b),
                ClauseOp48::SubRegReg(a, b) => self.v(a) - self.v(b),
                ClauseOp48::MinRegReg(a, b) => self.v(a).min(self.v(b)),
                ClauseOp48::MaxRegReg(a, b) => self.v(a).max(self.v(b)),
                ClauseOp48::AddRegImm(a, imm) => self.v(a) + imm,
                ClauseOp48::MulRegImm(a, imm) => self.v(a) * imm,
                ClauseOp48::SubImmReg(a, imm) => imm - self.v(a),
                ClauseOp48::SubRegImm(a, imm) => self.v(a) - imm,
                ClauseOp48::MinRegImm(a, imm) => self.v(a).min(imm),
                ClauseOp48::MaxRegImm(a, imm) => self.v(a).max(imm),
                ClauseOp48::CopyImm(imm) => imm,
            };
        }
        self.slots[self.slots.len() - 1]
    }
}

////////////////////////////////////////////////////////////////////////////////

struct TapeBuilder<'a> {
    iter: std::slice::Iter<'a, (NodeIndex, Op)>,
    vars: &'a IndexMap<String, VarIndex>,
    mapping: BTreeMap<NodeIndex, u32>,
    constants: BTreeMap<NodeIndex, f32>,
    choice_count: usize,

    /// `last_used[i]` is the last use of slot `i` during forward evaluation
    last_used: Vec<usize>,
}

enum Location {
    Register(u32),
    Immediate(f32),
}

impl<'a> TapeBuilder<'a> {
    fn new(t: &'a Scheduled) -> Self {
        Self {
            iter: t.tape.iter(),
            vars: &t.vars,
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
            last_used: vec![],
        }
    }

    fn get_allocated_value(&mut self, node: NodeIndex) -> Location {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Location::Register(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Location::Immediate(*c)
        }
    }

    fn run(&mut self) -> Vec<ClauseOp48> {
        self.collect()
    }

    fn step(&mut self, node: NodeIndex, op: Op) -> Option<ClauseOp48> {
        type RegRegFn = fn(u32, u32) -> ClauseOp48;
        type RegImmFn = fn(u32, f32) -> ClauseOp48;

        let index = self.mapping.len();
        assert_eq!(index, self.last_used.len());
        let out = match op {
            Op::Var(v) => {
                let arg = match self.vars.get_by_index(v).unwrap().as_str() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    _ => panic!(),
                };
                ClauseOp48::Input(arg)
            }
            Op::Const(c) => {
                // Skip this (because it's not inserted into the tape)
                // and recurse.  Hopefully, this is a tail call!
                self.constants.insert(node, c as f32);
                return None;
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);

                let f: (RegRegFn, RegImmFn, RegImmFn) = match op {
                    BinaryOpcode::Add => (
                        ClauseOp48::AddRegReg,
                        ClauseOp48::AddRegImm,
                        ClauseOp48::AddRegImm,
                    ),
                    BinaryOpcode::Mul => (
                        ClauseOp48::MulRegReg,
                        ClauseOp48::MulRegImm,
                        ClauseOp48::MulRegImm,
                    ),
                    BinaryOpcode::Sub => (
                        ClauseOp48::SubRegReg,
                        ClauseOp48::SubRegImm,
                        ClauseOp48::SubImmReg,
                    ),
                };

                match (lhs, rhs) {
                    (Location::Register(lhs), Location::Register(rhs)) => {
                        self.last_used[lhs as usize] = index;
                        self.last_used[rhs as usize] = index;
                        f.0(lhs, rhs)
                    }
                    (Location::Register(arg), Location::Immediate(imm)) => {
                        self.last_used[arg as usize] = index;
                        f.1(arg, imm)
                    }
                    (Location::Immediate(imm), Location::Register(arg)) => {
                        self.last_used[arg as usize] = index;
                        f.2(arg, imm)
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                }
            }
            Op::BinaryChoice(op, lhs, rhs, ..) => {
                self.choice_count += 1;
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);

                let f: (RegRegFn, RegImmFn) = match op {
                    BinaryChoiceOpcode::Min => {
                        (ClauseOp48::MinRegReg, ClauseOp48::MinRegImm)
                    }
                    BinaryChoiceOpcode::Max => {
                        (ClauseOp48::MaxRegReg, ClauseOp48::MaxRegImm)
                    }
                };

                match (lhs, rhs) {
                    (Location::Register(lhs), Location::Register(rhs)) => {
                        self.last_used[lhs as usize] = index;
                        self.last_used[rhs as usize] = index;
                        f.0(lhs, rhs)
                    }
                    (Location::Register(arg), Location::Immediate(imm)) => {
                        f.1(arg, imm)
                    }
                    (Location::Immediate(imm), Location::Register(arg)) => {
                        f.1(arg, imm)
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                }
            }
            Op::Unary(op, lhs) => {
                let lhs = match self.get_allocated_value(lhs) {
                    Location::Register(r) => r,
                    Location::Immediate(..) => {
                        panic!("Cannot handle f(imm)")
                    }
                };
                self.last_used[lhs as usize] = index;
                match op {
                    UnaryOpcode::Neg => ClauseOp48::NegReg(lhs),
                    UnaryOpcode::Abs => ClauseOp48::AbsReg(lhs),
                    UnaryOpcode::Recip => ClauseOp48::RecipReg(lhs),
                    UnaryOpcode::Sqrt => ClauseOp48::SqrtReg(lhs),
                    UnaryOpcode::Square => ClauseOp48::SquareReg(lhs),
                }
            }
        };
        let r = self.mapping.insert(node, index.try_into().unwrap());
        assert!(r.is_none());
        self.last_used.push(usize::MAX);
        Some(out)
    }
}

impl<'a> Iterator for TapeBuilder<'a> {
    type Item = ClauseOp48;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (n, op) = *self.iter.next()?;
            let r = self.step(n, op);
            if r.is_some() {
                break r;
            }
            // TODO: handle immediate-only tree
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Iterator-style... thingy which performs register allocation on a limited set
/// of registers, allowing for reuse.
pub struct TapeAllocator<'a> {
    /// Forward iterator over the tape
    iter:
        std::iter::Enumerate<std::iter::Rev<std::slice::Iter<'a, ClauseOp48>>>,

    /// Represents the last use of a node in the iterator
    last_use: &'a [usize],

    /// Map from the index in the original (globally allocated) tape to a
    /// specific register or memory slot.
    allocations: Vec<usize>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `usize::MAX` if the register is currently unused.
    registers: Vec<usize>,
    register_lru: Vec<usize>,
    time: usize,

    /// User-defined register limit; beyond this point we use load/store
    /// operations to move values to and from memory.
    reg_limit: usize,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_registers: Vec<usize>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_memory: Vec<usize>,

    /// Active nodes (as local slot indexes), sorted by position in
    /// the input data
    active: BTreeSet<(usize, usize)>,

    /// If we popped a clause but haven't finished it, store it here
    wip: Option<(usize, ClauseOp48)>,

    /// Total allocated slots
    total_slots: usize,
}

impl<'a> TapeAllocator<'a> {
    pub fn new(tape: &'a Tape, reg_limit: usize) -> Self {
        Self {
            iter: tape.tape.iter().rev().enumerate(),
            last_use: &tape.last_used,
            allocations: vec![usize::MAX; tape.tape.len()],

            registers: vec![usize::MAX; reg_limit],
            register_lru: vec![0; reg_limit],
            time: 0,

            reg_limit,
            spare_registers: Vec::with_capacity(reg_limit),
            spare_memory: vec![],
            wip: None,
            total_slots: 0,
            active: BTreeSet::new(),
        }
    }
    fn get_memory(&mut self) -> usize {
        if let Some(p) = self.spare_memory.pop() {
            p
        } else {
            let out = self.total_slots;
            self.total_slots += 1;
            out
        }
    }

    fn oldest_reg(&self) -> usize {
        self.register_lru
            .iter()
            .enumerate()
            .min_by_key(|i| i.1)
            .unwrap()
            .0
    }

    /// Looks up a node by global index
    ///
    /// The node must already be in a register, otherwise this will panic
    fn get(&mut self, n: u32) -> u32 {
        let slot = self.allocations[n as usize];
        assert!(slot < self.reg_limit);
        slot as u32
    }

    /// Attempt to get a register for the given node (which is an index into the
    /// globally-allocated tape).
    ///
    /// This happens to work if a node is unassigned, because at that point, it
    /// will be allocated to `usize::MAX`, which looks like a (very far away)
    /// slot in memory.
    ///
    /// This may take multiple attempts, returning an `Err(CopyOp::...)`
    /// when intermediate memory movement needs to take place.
    fn get_register(&mut self, n: usize) -> Result<usize, CopyOp> {
        let slot = self.allocations[n];
        if slot >= self.reg_limit {
            // Pick a register, prioritizing picking a spare register (if
            // possible); if not, then introducing a new register (if we haven't
            // allocated past our limit)
            let reg = self.spare_registers.pop().or_else(|| {
                if self.total_slots < self.reg_limit {
                    let reg = self.total_slots;
                    self.total_slots += 1;
                    Some(reg)
                } else {
                    self.spare_registers.pop()
                }
            });

            if let Some(reg) = reg {
                // If we've got a spare register, then we can use it by adding a
                // `Load` instruction to the stream.
                assert_eq!(self.registers[reg], usize::MAX);

                self.registers[reg] = n;
                self.allocations[n] = reg;

                // Release the memory slot that we were previously using, if
                // it's not the dummy slot indicating no assignment was made.
                if slot == usize::MAX {
                    self.register_lru[reg] = self.time;
                    Ok(reg)
                } else {
                    self.spare_memory.push(slot);
                    Err(CopyOp {
                        src: slot as u32,
                        dst: reg as u32,
                    })
                }
            } else {
                // Otherwise, we need to free up a register by pushing the
                // oldest value to a slot in memory.
                let mem = self.get_memory();
                let reg = self.oldest_reg();

                // Whoever was previously using you is in for a surprise
                let prev_node = self.registers[reg];
                self.allocations[prev_node] = mem;

                // Be free, young register!
                self.registers[reg] = usize::MAX;
                self.spare_registers.push(reg);

                // (next time we pass this way, the register will be available
                //  in self.spare_registers!)

                Err(CopyOp {
                    src: reg as u32,
                    dst: mem as u32,
                })
            }
        } else {
            // Update the use time of this register
            self.register_lru[slot] = self.time;
            Ok(slot)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CopyOp {
    src: u32,
    dst: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct AllocOp(ClauseOp48, u32);

impl<'a> Iterator for TapeAllocator<'a> {
    type Item = AllocOp;
    fn next(&mut self) -> Option<Self::Item> {
        let (index, op): (usize, ClauseOp48) = self
            .wip
            .or_else(|| self.iter.next().map(|(i, n)| (i, *n)))?;

        self.time += 1;

        // Make sure that we have LHS and RHS already loaded into registers,
        // returning early with a Load or Store if necessary.
        match op {
            ClauseOp48::Input(..) | ClauseOp48::CopyImm(..) => (),
            ClauseOp48::NegReg(arg)
            | ClauseOp48::AbsReg(arg)
            | ClauseOp48::RecipReg(arg)
            | ClauseOp48::CopyReg(arg)
            | ClauseOp48::SqrtReg(arg)
            | ClauseOp48::SquareReg(arg)
            | ClauseOp48::AddRegImm(arg, ..)
            | ClauseOp48::MulRegImm(arg, ..)
            | ClauseOp48::SubImmReg(arg, ..)
            | ClauseOp48::SubRegImm(arg, ..)
            | ClauseOp48::MinRegImm(arg, ..)
            | ClauseOp48::MaxRegImm(arg, ..) => {
                match self.get_register(arg as usize) {
                    Ok(_reg) => (),
                    Err(CopyOp { src, dst }) => {
                        self.wip = Some((index, op));
                        return Some(AllocOp(ClauseOp48::CopyReg(src), dst));
                    }
                }
            }

            ClauseOp48::AddRegReg(lhs, rhs)
            | ClauseOp48::MulRegReg(lhs, rhs)
            | ClauseOp48::SubRegReg(lhs, rhs)
            | ClauseOp48::MinRegReg(lhs, rhs)
            | ClauseOp48::MaxRegReg(lhs, rhs) => {
                match self.get_register(lhs as usize) {
                    Ok(_reg) => (),
                    Err(CopyOp { src, dst }) => {
                        self.wip = Some((index, op));
                        return Some(AllocOp(ClauseOp48::CopyReg(src), dst));
                    }
                }
                match self.get_register(rhs as usize) {
                    Ok(_reg) => (),
                    Err(CopyOp { src, dst }) => {
                        self.wip = Some((index, op));
                        return Some(AllocOp(ClauseOp48::CopyReg(src), dst));
                    }
                }
            }
        }

        // Release anything that's inactive at this point, so that the output
        // could reuse a register from one of the inputs if this is the last
        // time it appears.
        while let Some((j, node)) = self.active.iter().next().cloned() {
            if j >= index {
                break;
            }
            self.active.remove(&(j, node));
            let slot = self.allocations[node];
            if slot < self.reg_limit {
                self.spare_registers.push(slot);
                self.registers[slot] = usize::MAX;
                self.register_lru[slot] = usize::MAX;
                // Note that this leaves self.allocations[node] still pointing
                // to the old register, but that's okay, because it should never
                // be used again!
            } else {
                self.spare_memory.push(slot);
            }
        }

        let out_reg = match self.get_register(index) {
            Ok(reg) => reg,
            Err(CopyOp { src, dst }) => {
                self.wip = Some((index, op));
                return Some(AllocOp(ClauseOp48::CopyReg(src), dst));
            }
        };

        use ClauseOp48::*;
        let out_op = match op {
            Input(..) | CopyImm(..) => op,
            AddRegReg(lhs, rhs) => AddRegReg(self.get(lhs), self.get(rhs)),
            MulRegReg(lhs, rhs) => MulRegReg(self.get(lhs), self.get(rhs)),
            SubRegReg(lhs, rhs) => SubRegReg(self.get(lhs), self.get(rhs)),
            NegReg(arg) => NegReg(self.get(arg)),
            CopyReg(arg) => CopyReg(self.get(arg)),
            AbsReg(arg) => AbsReg(self.get(arg)),
            RecipReg(arg) => RecipReg(self.get(arg)),
            SqrtReg(arg) => SqrtReg(self.get(arg)),
            SquareReg(arg) => SquareReg(self.get(arg)),

            AddRegImm(arg, imm) => AddRegImm(self.get(arg), imm),
            MulRegImm(arg, imm) => MulRegImm(self.get(arg), imm),
            SubImmReg(arg, imm) => SubImmReg(self.get(arg), imm),
            SubRegImm(arg, imm) => SubRegImm(self.get(arg), imm),

            MinRegImm(arg, imm) => MinRegImm(self.get(arg), imm),
            MaxRegImm(arg, imm) => MaxRegImm(self.get(arg), imm),
            MinRegReg(lhs, rhs) => MinRegReg(self.get(lhs), self.get(rhs)),
            MaxRegReg(lhs, rhs) => MaxRegReg(self.get(lhs), self.get(rhs)),
        };

        self.active.insert((self.last_use[index], index));

        // If we've gotten here, then we've cleared the WIP node and are about
        // to deliver some actual output.
        self.wip = None;

        Some(AllocOp(out_op, out_reg.try_into().unwrap()))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct AllocatedTape {
    /// Raw instruction tape, stored in forward (evaluation) order
    pub tape: Vec<AllocOp>,

    pub total_slots: usize,
    pub out_slot: usize,

    /// The number of nodes which store values in the choice array during
    /// interval evaluation.
    pub choice_count: usize,
}

impl AllocatedTape {
    /// Builds an evaluator which takes a (read-only) reference to this tape
    pub fn get_evaluator(&self) -> AllocatedTapeEval {
        AllocatedTapeEval {
            tape: self,
            slots: vec![0.0; self.total_slots],
            out_slot: self.out_slot,
        }
    }
}

/// Workspace to evaluate a tape
pub struct AllocatedTapeEval<'a> {
    tape: &'a AllocatedTape,
    out_slot: usize,
    slots: Vec<f32>,
}

impl<'a> AllocatedTapeEval<'a> {
    fn v(&self, i: u32) -> f32 {
        self.slots[i as usize]
    }
    pub fn f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        use ClauseOp48::*;

        for &AllocOp(op, out) in self.tape.tape.iter() {
            self.slots[out as usize] = match op {
                Input(i) => match i {
                    0 => x,
                    1 => y,
                    2 => z,
                    _ => panic!(),
                },
                NegReg(i) => -self.v(i),
                AbsReg(i) => self.v(i).abs(),
                RecipReg(i) => 1.0 / self.v(i),
                SqrtReg(i) => self.v(i).sqrt(),
                SquareReg(i) => self.v(i) * self.v(i),

                AddRegReg(a, b) => self.v(a) + self.v(b),
                MulRegReg(a, b) => self.v(a) * self.v(b),
                SubRegReg(a, b) => self.v(a) - self.v(b),
                MinRegReg(a, b) => self.v(a).min(self.v(b)),
                MaxRegReg(a, b) => self.v(a).max(self.v(b)),
                AddRegImm(a, imm) => self.v(a) + imm,
                MulRegImm(a, imm) => self.v(a) * imm,
                SubImmReg(a, imm) => imm - self.v(a),
                SubRegImm(a, imm) => self.v(a) - imm,
                MinRegImm(a, imm) => self.v(a).min(imm),
                MaxRegImm(a, imm) => self.v(a).max(imm),
                CopyImm(imm) => imm,

                CopyReg(arg) => self.v(arg),
            };
        }
        self.slots[self.out_slot]
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::common::Choice;

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
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 3.5, 0.0), 3.5);
    }

    #[test]
    fn test_push() {
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn test_alloc() {
        dbg!(std::mem::size_of::<ClauseOp48>());
        dbg!(std::mem::size_of::<AllocOp>());
        dbg!(std::mem::size_of::<(ClauseOp48, u32)>());
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let lol = tape.alloc(3);
        let mut eval = lol.get_evaluator();
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 4.0, 0.0), 3.0);
    }
}
