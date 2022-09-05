use std::collections::BTreeMap;

use crate::scheduled::Scheduled;
use crate::{
    backend::common::{Choice, NodeIndex, Op},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
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
        builder.run();
        builder.out.reverse();
        Self {
            tape: builder.out,
            choice_count: builder.choice_count,
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
        let mut out = vec![];
        let mut choice_count = 0;
        let mut choice_iter = choices.iter();
        let mut remap = Vec::with_capacity(self.tape.len());

        for (index, op) in self.tape.iter().rev().cloned().enumerate() {
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
                remap.push(u32::MAX);
                continue;
            }

            let op = match op {
                Input(..) | CopyImm(..) => op,
                AddRegReg(lhs, rhs) => {
                    AddRegReg(remap[lhs as usize], remap[rhs as usize])
                }
                MulRegReg(lhs, rhs) => {
                    MulRegReg(remap[lhs as usize], remap[rhs as usize])
                }
                SubRegReg(lhs, rhs) => {
                    SubRegReg(remap[lhs as usize], remap[rhs as usize])
                }
                NegReg(arg) => NegReg(remap[arg as usize]),
                AbsReg(arg) => AbsReg(remap[arg as usize]),
                RecipReg(arg) => RecipReg(remap[arg as usize]),
                SqrtReg(arg) => SqrtReg(remap[arg as usize]),
                SquareReg(arg) => SquareReg(remap[arg as usize]),

                AddRegImm(arg, imm) => AddRegImm(remap[arg as usize], imm),
                MulRegImm(arg, imm) => MulRegImm(remap[arg as usize], imm),
                SubImmReg(arg, imm) => SubImmReg(remap[arg as usize], imm),
                SubRegImm(arg, imm) => SubRegImm(remap[arg as usize], imm),

                MinRegImm(arg, imm) | MaxRegImm(arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            remap.push(remap[arg as usize]);
                            continue;
                        }
                        Choice::Right => CopyImm(imm),
                        Choice::Both => {
                            choice_count += 1;
                            match op {
                                MinRegImm(arg, imm) => {
                                    MinRegImm(remap[arg as usize], imm)
                                }
                                MaxRegImm(arg, imm) => {
                                    MaxRegImm(remap[arg as usize], imm)
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                MinRegReg(lhs, rhs) | MaxRegReg(lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            remap.push(remap[lhs as usize]);
                            continue;
                        }
                        Choice::Right => {
                            remap.push(remap[rhs as usize]);
                            continue;
                        }
                        Choice::Both => {
                            choice_count += 1;
                            match op {
                                MinRegReg(lhs, rhs) => MinRegReg(
                                    remap[lhs as usize],
                                    remap[rhs as usize],
                                ),
                                MaxRegReg(lhs, rhs) => MaxRegReg(
                                    remap[lhs as usize],
                                    remap[rhs as usize],
                                ),
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            };
            remap.push(out.len() as u32);
            out.push(op);
        }

        Self {
            tape: out,
            choice_count,
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
    t: &'a Scheduled,
    out: Vec<ClauseOp48>,
    mapping: BTreeMap<NodeIndex, u32>,
    constants: BTreeMap<NodeIndex, f32>,
    choice_count: usize,
}

enum Allocation {
    Register(u32),
    Immediate(f32),
}

impl<'a> TapeBuilder<'a> {
    fn new(t: &'a Scheduled) -> Self {
        Self {
            t,
            out: vec![],
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
        }
    }

    fn push(&mut self, node: NodeIndex, op: ClauseOp48) {
        let r = self
            .mapping
            .insert(node, self.out.len().try_into().unwrap());
        assert!(r.is_none());
        self.out.push(op);
    }

    fn get_allocated_value(&mut self, node: NodeIndex) -> Allocation {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Allocation::Register(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Allocation::Immediate(*c)
        }
    }

    fn run(&mut self) {
        type RegRegFn = fn(u32, u32) -> ClauseOp48;
        type RegImmFn = fn(u32, f32) -> ClauseOp48;

        for &(n, op) in &self.t.tape {
            let out = match op {
                Op::Var(v) => {
                    let index =
                        match self.t.vars.get_by_index(v).unwrap().as_str() {
                            "X" => 0,
                            "Y" => 1,
                            "Z" => 2,
                            _ => panic!(),
                        };
                    ClauseOp48::Input(index)
                }
                Op::Const(c) => {
                    self.constants.insert(n, c as f32);
                    continue;
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
                        (
                            Allocation::Register(lhs),
                            Allocation::Register(rhs),
                        ) => f.0(lhs, rhs),
                        (
                            Allocation::Register(arg),
                            Allocation::Immediate(imm),
                        ) => f.1(arg, imm),
                        (
                            Allocation::Immediate(imm),
                            Allocation::Register(arg),
                        ) => f.2(arg, imm),
                        _ => panic!("Cannot handle f(imm, imm)"),
                    }
                    // TODO
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
                        (
                            Allocation::Register(lhs),
                            Allocation::Register(rhs),
                        ) => f.0(lhs, rhs),
                        (
                            Allocation::Register(arg),
                            Allocation::Immediate(imm),
                        ) => f.1(arg, imm),
                        (
                            Allocation::Immediate(imm),
                            Allocation::Register(arg),
                        ) => f.1(arg, imm),
                        _ => panic!("Cannot handle f(imm, imm)"),
                    }
                }
                Op::Unary(op, lhs) => {
                    let lhs = match self.get_allocated_value(lhs) {
                        Allocation::Register(r) => r,
                        Allocation::Immediate(..) => {
                            panic!("Cannot handle f(imm)")
                        }
                    };
                    match op {
                        UnaryOpcode::Neg => ClauseOp48::NegReg(lhs),
                        UnaryOpcode::Abs => ClauseOp48::AbsReg(lhs),
                        UnaryOpcode::Recip => ClauseOp48::RecipReg(lhs),
                        UnaryOpcode::Sqrt => ClauseOp48::SqrtReg(lhs),
                        UnaryOpcode::Square => ClauseOp48::SquareReg(lhs),
                    }
                }
            };
            self.push(n, out);
        }
    }
}

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
