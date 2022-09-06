use std::collections::BTreeMap;

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
    /// Raw instruction tape
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
            AbsReg(arg) => AbsReg(self.get(arg)),
            RecipReg(arg) => RecipReg(self.get(arg)),
            SqrtReg(arg) => {
                self.last_used[arg as usize] = index;
                SqrtReg(self.get(arg))
            }
            SquareReg(arg) => {
                self.last_used[arg as usize] = index;
                SquareReg(self.get(arg))
            }

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
    pub fn f(&mut self, x: f32, y: f32) -> f32 {
        for (i, &op) in self.tape.tape.iter().rev().enumerate() {
            self.slots[i] = match op {
                ClauseOp48::Input(i) => match i {
                    0 => x,
                    1 => y,
                    _ => panic!(),
                },
                ClauseOp48::NegReg(i) => -self.v(i),
                ClauseOp48::AbsReg(i) => self.v(i).abs(),
                ClauseOp48::RecipReg(i) => 1.0 / self.v(i),
                ClauseOp48::SqrtReg(i) => self.v(i).sqrt(),
                ClauseOp48::SquareReg(i) => self.v(i) * self.v(i),

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

enum Allocation {
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

    fn get_allocated_value(&mut self, node: NodeIndex) -> Allocation {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Allocation::Register(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Allocation::Immediate(*c)
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
                    (Allocation::Register(lhs), Allocation::Register(rhs)) => {
                        self.last_used[lhs as usize] = index;
                        self.last_used[rhs as usize] = index;
                        f.0(lhs, rhs)
                    }
                    (Allocation::Register(arg), Allocation::Immediate(imm)) => {
                        self.last_used[arg as usize] = index;
                        f.1(arg, imm)
                    }
                    (Allocation::Immediate(imm), Allocation::Register(arg)) => {
                        self.last_used[arg as usize] = index;
                        f.2(arg, imm)
                    }
                    (Allocation::Immediate(..), Allocation::Immediate(..)) => {
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
                    (Allocation::Register(lhs), Allocation::Register(rhs)) => {
                        self.last_used[lhs as usize] = index;
                        self.last_used[rhs as usize] = index;
                        f.0(lhs, rhs)
                    }
                    (Allocation::Register(arg), Allocation::Immediate(imm)) => {
                        f.1(arg, imm)
                    }
                    (Allocation::Immediate(imm), Allocation::Register(arg)) => {
                        f.1(arg, imm)
                    }
                    (Allocation::Immediate(..), Allocation::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                }
            }
            Op::Unary(op, lhs) => {
                let lhs = match self.get_allocated_value(lhs) {
                    Allocation::Register(r) => r,
                    Allocation::Immediate(..) => {
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
    iter: std::slice::Iter<'a, ClauseOp48>,

    /// Map from the index in the original (globally allocated) tape to a
    /// specific register or memory slot.
    allocations: Vec<u32>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `u32::MAX` if the register is currently unused.
    registers: Vec<u32>,
    register_lru: Vec<u32>,
    time: u32,

    /// User-defined register limit; beyond this point we use load/store
    /// operations to move values to and from memory.
    reg_limit: u8,

    /// Available short registers (index < 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_registers: Vec<u32>,

    /// Available extended registers (index >= 256)
    ///
    /// The most recently available is at the back of the `Vec`
    spare_memory: Vec<u32>,

    /// If we popped a clause but haven't finished it, store it here
    wip: Option<ClauseOp48>,

    /// Total allocated slots
    total_slots: u32,
}

impl<'a> TapeAllocator<'a> {
    pub fn new(tape: &'a Tape, reg_limit: u8) -> Self {
        Self {
            iter: tape.tape.iter(),
            allocations: vec![u32::MAX; tape.tape.len()],

            registers: vec![u32::MAX; reg_limit as usize],
            register_lru: vec![0; reg_limit as usize],
            time: 0,

            reg_limit,
            spare_registers: Vec::with_capacity(reg_limit as usize),
            spare_memory: (0..reg_limit).map(|i| i as u32).collect(),
            wip: None,
            total_slots: reg_limit as u32,
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

    fn oldest_reg(&self) -> u32 {
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
    /// The given node must have been assigned.
    ///
    /// This may take multiple attempts, returning an `Err(LoadStoreOp::...)`
    /// when intermediate memory movement needs to take place.
    fn get_register(&mut self, n: u32) -> Result<u32, LoadStoreOp> {
        let slot = self.allocations[n as usize];
        if slot >= self.reg_limit as u32 {
            if let Some(reg) = self.spare_registers.pop() {
                // If we've got a spare register, then we can use it by adding a
                // `Load` instruction to the stream.
                assert_eq!(self.registers[reg as usize], u32::MAX);

                // Release the memory slot that we were previously using
                self.spare_memory.push(slot);

                self.registers[reg as usize] = n;
                self.allocations[n as usize] = reg;
                Err(LoadStoreOp::Load {
                    src: slot,
                    dst: reg,
                })
            } else {
                // Otherwise, we need to free up a register by pushing the
                // oldest value to a slot in memory.
                let mem = self.get_memory();
                let reg = self.oldest_reg();
                let prev_node = self.registers[reg as usize];
                self.registers[reg as usize] = u32::MAX;
                self.spare_registers.push(reg);
                self.allocations[prev_node as usize] = mem;
                Err(LoadStoreOp::Store { src: reg, dst: mem })
            }
        } else {
            // Update the use time of this register
            self.register_lru[slot as usize] = self.time;
            Ok(slot)
        }
    }
}

pub enum LoadStoreOp {
    Load { src: u32, dst: u32 },
    Store { src: u32, dst: u32 },
}

pub enum AllocOp {
    LoadStore(LoadStoreOp),
    Op(ClauseOp48),
}

impl<'a> Iterator for TapeAllocator<'a> {
    type Item = AllocOp;
    fn next(&mut self) -> Option<Self::Item> {
        let op: Option<ClauseOp48> =
            self.wip.or_else(|| self.iter.next().cloned());

        self.time += 1;

        // Check if LHS is available
        //      If not, check if a register is available
        //          Yes => Emit a LOAD
        //          No =>, boot a register into memory, emitting a STORE
        // Check if RHS is available

        // Every clause in the original tape expands to one or more clauses.
        todo!()
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
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0), 1.0);
    }
}
