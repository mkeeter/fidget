use std::collections::BTreeMap;

use crate::scheduled::Scheduled;
use crate::{
    backend::common::{NodeIndex, Op},
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

    AddRegReg(u32, u32),
    MulRegReg(u32, u32),
    SubRegReg(u32, u32),
    MinRegReg(u32, u32),
    MaxRegReg(u32, u32),

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
