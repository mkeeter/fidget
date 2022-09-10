use std::collections::BTreeMap;

use crate::scheduled::Scheduled;
use crate::{
    backend::{
        common::{Choice, NodeIndex, Op, Simplify, VarIndex},
        dynasm::AsmOp,
    },
    op::{BinaryOpcode, UnaryOpcode},
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

/// Tape storing 48-bit (12-byte) operations (TODO check this math):
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
}

impl Simplify for Tape {
    fn simplify(&self, choices: &[Choice]) -> Self {
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
                        Choice::Unknown => {
                            panic!("oh no")
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
                        Choice::Unknown => {
                            panic!("oh no")
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
            remap: Vec::with_capacity(self.tape.len()),
            last_used: vec![],
        };

        let mut out = Vec::with_capacity(self.tape.len());
        out.extend(&mut simplify);
        out.reverse();

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
    // TODO: we could track an exact size here and implement `ExactSizeIterator`
}

impl<'a, I> TapeSimplify<'a, I>
where
    I: Iterator<Item = ClauseOp48>,
{
    fn get(&mut self, i: u32) -> u32 {
        let r = self.remap[i as usize];
        self.last_used[r as usize] = self.last_used.len();
        r
    }

    fn step(&mut self, mut op: ClauseOp48) -> Option<ClauseOp48> {
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

        match &mut op {
            Input(..) | CopyImm(..) => (),
            AddRegReg(lhs, rhs) | MulRegReg(lhs, rhs) | SubRegReg(lhs, rhs) => {
                *lhs = self.get(*lhs);
                *rhs = self.get(*rhs);
            }
            NegReg(arg) | CopyReg(arg) | AbsReg(arg) | RecipReg(arg)
            | SqrtReg(arg) | SquareReg(arg) => {
                *arg = self.get(*arg);
            }

            AddRegImm(arg, ..)
            | MulRegImm(arg, ..)
            | SubImmReg(arg, ..)
            | SubRegImm(arg, ..) => {
                *arg = self.get(*arg);
            }

            MinRegImm(arg, imm) | MaxRegImm(arg, imm) => {
                match self.choice_iter.next().unwrap() {
                    Choice::Left => {
                        self.remap.push(self.remap[*arg as usize]);
                        return None;
                    }
                    Choice::Right => op = CopyImm(*imm),
                    Choice::Both => {
                        self.choice_count += 1;
                        *arg = self.get(*arg);
                    }
                    Choice::Unknown => {
                        panic!("oh no")
                    }
                }
            }
            MinRegReg(lhs, rhs) | MaxRegReg(lhs, rhs) => {
                match self.choice_iter.next().unwrap() {
                    Choice::Left => {
                        self.remap.push(self.remap[*lhs as usize]);
                        return None;
                    }
                    Choice::Right => {
                        self.remap.push(self.remap[*rhs as usize]);
                        return None;
                    }
                    Choice::Both => {
                        self.choice_count += 1;
                        *lhs = self.get(*lhs);
                        *rhs = self.get(*rhs);
                    }
                    Choice::Unknown => {
                        panic!("oh no")
                    }
                }
            }
        };
        self.remap.push(self.last_used.len() as u32);
        self.last_used.push(usize::MAX);
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
                    BinaryOpcode::Min => (
                        ClauseOp48::MinRegReg,
                        ClauseOp48::MinRegImm,
                        ClauseOp48::MinRegImm,
                    ),
                    BinaryOpcode::Max => (
                        ClauseOp48::MaxRegReg,
                        ClauseOp48::MaxRegImm,
                        ClauseOp48::MaxRegImm,
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
    allocations: Vec<u32>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `usize::MAX` if the register is currently unused.
    registers: Vec<usize>,

    /// For reach register, this `Vec` stores its last access time
    register_lru: Vec<usize>,
    time: usize,

    /// User-defined register limit; beyond this point we use load/store
    /// operations to move values to and from memory.
    reg_limit: u8,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_registers: Vec<u8>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_memory: Vec<u32>,

    /// Represents which nodes retire at a particular index in the global tape.
    /// Up to two nodes can be retired at each index; `usize::MAX` indicates
    /// that there isn't a retired node here yet.
    retirement: Vec<[usize; 2]>,

    /// If we popped a clause but haven't finished it, store it here
    wip: Option<(usize, ClauseOp48)>,

    /// Total allocated slots
    total_slots: u32,

    /// Have we inserted a final copy operation into the tape?
    done: bool,
}

impl<'a> TapeAllocator<'a> {
    pub fn new(tape: &'a Tape, reg_limit: u8) -> Self {
        Self {
            iter: tape.tape.iter().rev().enumerate(),
            last_use: &tape.last_used,
            allocations: vec![u32::MAX; tape.tape.len()],

            registers: vec![usize::MAX; reg_limit as usize],
            register_lru: vec![0; reg_limit as usize],
            time: 0,

            retirement: vec![[usize::MAX; 2]; tape.tape.len()],
            reg_limit,
            spare_registers: Vec::with_capacity(reg_limit as usize),
            spare_memory: vec![],
            wip: None,
            total_slots: 0,
            done: false,
        }
    }
    fn get_memory(&mut self) -> u32 {
        if let Some(p) = self.spare_memory.pop() {
            p
        } else {
            let out = self.total_slots;
            self.total_slots += 1;
            out
        }
    }

    fn oldest_reg(&self) -> u8 {
        self.register_lru
            .iter()
            .enumerate()
            .min_by_key(|i| i.1)
            .unwrap()
            .0
            .try_into()
            .unwrap()
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
    fn get_register(&mut self, n: u32) -> Result<u8, CopyOp> {
        let n = n as usize;
        let slot = self.allocations[n];
        if slot >= self.reg_limit as u32 {
            // Pick a register, prioritizing picking a spare register (if
            // possible); if not, then introducing a new register (if we haven't
            // allocated past our limit)
            let reg = self.spare_registers.pop().or_else(|| {
                if self.total_slots < self.reg_limit as u32 {
                    let reg = self.total_slots;
                    self.total_slots += 1;
                    Some(reg.try_into().unwrap())
                } else {
                    None
                }
            });

            if let Some(reg) = reg {
                // If we've got a spare register, then we can use it by adding a
                // `Load` instruction to the stream.
                assert_eq!(self.registers[reg as usize], usize::MAX);

                self.registers[reg as usize] = n;
                self.allocations[n] = reg as u32;

                // Release the memory slot that we were previously using, if
                // it's not the dummy slot indicating no assignment was made.
                if slot == u32::MAX {
                    self.register_lru[reg as usize] = self.time;
                    Ok(reg as u8)
                } else {
                    self.spare_memory.push(slot);
                    Err(CopyOp::Load {
                        src: slot,
                        dst: reg,
                    })
                }
            } else {
                // Otherwise, we need to free up a register by pushing the
                // oldest value to a slot in memory.
                let mem = self.get_memory();
                let reg = self.oldest_reg();

                // Whoever was previously using you is in for a surprise
                let prev_node = self.registers[reg as usize];
                self.allocations[prev_node] = mem;

                // Be free, young register!
                self.registers[reg as usize] = usize::MAX;
                self.spare_registers.push(reg);

                // (next time we pass this way, the register will be available
                //  in self.spare_registers!)

                Err(CopyOp::Store { src: reg, dst: mem })
            }
        } else {
            // Update the use time of this register
            self.register_lru[slot as usize] = self.time;
            Ok(slot as u8)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CopyOp {
    Load { src: u32, dst: u8 },
    Store { src: u8, dst: u32 },
}

impl From<CopyOp> for AsmOp {
    fn from(c: CopyOp) -> Self {
        match c {
            CopyOp::Load { src, dst } => AsmOp::Load(dst, src),
            CopyOp::Store { src, dst } => AsmOp::Store(src, dst),
        }
    }
}

impl<'a> Iterator for TapeAllocator<'a> {
    type Item = AsmOp;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self
            .wip
            .take()
            .or_else(|| self.iter.next().map(|(i, n)| (i, *n)));

        let (index, op) = match next {
            Some((index, op)) => (index, op),
            None if !self.done => {
                // Copy from the last write into register 0
                self.done = true;
                let last_reg = *self.allocations.last().unwrap();
                if last_reg > 0 {
                    assert!(last_reg < self.reg_limit as u32);
                    return Some(AsmOp::CopyReg(0, last_reg as u8));
                } else {
                    return None;
                }
            }
            None => return None,
        };

        self.time += 1;

        // Make sure that we have LHS and RHS already loaded into registers,
        // returning early with a Load or Store if necessary.
        let (lhs, rhs) = match op {
            ClauseOp48::Input(..) | ClauseOp48::CopyImm(..) => (0, 0),
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
            | ClauseOp48::MaxRegImm(arg, ..) => match self.get_register(arg) {
                Err(c) => {
                    self.wip = Some((index, op));
                    return Some(AsmOp::from(c));
                }
                Ok(r) => (r, 0),
            },

            ClauseOp48::AddRegReg(lhs, rhs)
            | ClauseOp48::MulRegReg(lhs, rhs)
            | ClauseOp48::SubRegReg(lhs, rhs)
            | ClauseOp48::MinRegReg(lhs, rhs)
            | ClauseOp48::MaxRegReg(lhs, rhs) => {
                let lhs = match self.get_register(lhs) {
                    Err(c) => {
                        self.wip = Some((index, op));
                        return Some(AsmOp::from(c));
                    }
                    Ok(reg) => reg,
                };
                let rhs = match self.get_register(rhs) {
                    Err(c) => {
                        self.wip = Some((index, op));
                        return Some(AsmOp::from(c));
                    }
                    Ok(reg) => reg,
                };
                (lhs, rhs)
            }
        };

        // Release anything that's inactive at this point, so that the output
        // could reuse a register from one of the inputs if this is the last
        // time it appears.
        for node in self.retirement[index]
            .iter()
            .cloned()
            .filter(|i| *i != usize::MAX)
        {
            let slot = self.allocations[node];
            if slot < self.reg_limit as u32 {
                self.spare_registers.push(slot as u8);
                self.registers[slot as usize] = usize::MAX;
                self.register_lru[slot as usize] = usize::MAX;
                // Note that this leaves self.allocations[node] still pointing
                // to the old register, but that's okay, because it should never
                // be used again!
            } else {
                self.spare_memory.push(slot);
            }
        }

        let out_reg = match self.get_register(index.try_into().unwrap()) {
            Ok(reg) => reg,
            Err(c) => {
                self.wip = Some((index, op));
                return Some(AsmOp::from(c));
            }
        };

        use ClauseOp48::*;
        let out_op = match op {
            Input(i) => AsmOp::Input(out_reg, i),
            CopyImm(f) => AsmOp::CopyImm(out_reg, f),
            AddRegReg(..) => AsmOp::AddRegReg(out_reg, lhs, rhs),
            MulRegReg(..) => AsmOp::MulRegReg(out_reg, lhs, rhs),
            SubRegReg(..) => AsmOp::SubRegReg(out_reg, lhs, rhs),
            NegReg(..) => AsmOp::NegReg(out_reg, lhs),
            CopyReg(..) => AsmOp::CopyReg(out_reg, lhs),
            AbsReg(..) => AsmOp::AbsReg(out_reg, lhs),
            RecipReg(..) => AsmOp::RecipReg(out_reg, lhs),
            SqrtReg(..) => AsmOp::SqrtReg(out_reg, lhs),
            SquareReg(..) => AsmOp::SquareReg(out_reg, lhs),

            AddRegImm(_arg, imm) => AsmOp::AddRegImm(out_reg, lhs, imm),
            MulRegImm(_arg, imm) => AsmOp::MulRegImm(out_reg, lhs, imm),
            SubImmReg(_arg, imm) => AsmOp::SubImmReg(out_reg, lhs, imm),
            SubRegImm(_arg, imm) => AsmOp::SubRegImm(out_reg, lhs, imm),

            MinRegImm(_arg, imm) => AsmOp::MinRegImm(out_reg, lhs, imm),
            MaxRegImm(_arg, imm) => AsmOp::MaxRegImm(out_reg, lhs, imm),
            MinRegReg(..) => AsmOp::MinRegReg(out_reg, lhs, rhs),
            MaxRegReg(..) => AsmOp::MaxRegReg(out_reg, lhs, rhs),
        };

        // Install this node into the retirement array based on its last use
        if self.last_use[index] != usize::MAX {
            *self.retirement[self.last_use[index]]
                .iter_mut()
                .find(|i| **i == usize::MAX)
                .unwrap() = index;
        }

        Some(out_op)
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
}
