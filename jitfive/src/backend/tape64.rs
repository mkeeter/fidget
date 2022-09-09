use crate::{
    backend::common::{NodeIndex, Op, VarIndex},
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    scheduled::Scheduled,
    util::indexed::IndexMap,
};

use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug)]
pub enum ClauseOp64 {
    /// Reads one of the inputs (X, Y, Z)
    Input,

    NegReg,
    AbsReg,
    RecipReg,
    SqrtReg,
    SquareReg,

    /// Copies the given register
    CopyReg,

    /// Add a register and an immediate
    AddRegImm,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Subtract a register from an immediate
    SubImmReg,
    /// Subtract an immediate from a register
    SubRegImm,
    /// Compute the minimum of a register and an immediate
    MinRegImm,
    /// Compute the maximum of a register and an immediate
    MaxRegImm,

    AddRegReg,
    MulRegReg,
    SubRegReg,
    MinRegReg,
    MaxRegReg,

    /// Copy an immediate to a register
    CopyImm,
}

/// Tape storing... stuff
/// - 4-byte opcode
/// - 4-byte output register
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs, arguments, and immediates are packed into the `data` array
///
/// All slot addressing is absolute.
#[derive(Clone, Debug)]
pub struct Tape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<ClauseOp64>,

    /// Variable-length data for tape clauses.
    ///
    /// Data is densely packed in the order
    /// - output slot
    /// - lhs slot (or input)
    /// - rhs slot (or immediate)
    ///
    /// i.e. a unary operation would only store two items in this array
    pub data: Vec<u32>,

    /// Number of choice operations in the tape
    pub choice_count: usize,
}

impl Tape {
    pub fn new(t: &Scheduled) -> Self {
        let mut builder = TapeBuilder::new(t);
        builder.run();
        builder.tape.reverse();
        builder.data.reverse();
        Self {
            tape: builder.tape,
            data: builder.data,
            choice_count: builder.choice_count,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct TapeBuilder<'a> {
    iter: std::slice::Iter<'a, (NodeIndex, Op)>,

    tape: Vec<ClauseOp64>,
    data: Vec<u32>,

    vars: &'a IndexMap<String, VarIndex>,
    mapping: BTreeMap<NodeIndex, u32>,
    constants: BTreeMap<NodeIndex, f32>,
    choice_count: usize,
}

enum Location {
    Register(u32),
    Immediate(f32),
}

impl<'a> TapeBuilder<'a> {
    fn new(t: &'a Scheduled) -> Self {
        Self {
            iter: t.tape.iter(),
            tape: vec![],
            data: vec![],
            vars: &t.vars,
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
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

    fn run(&mut self) -> Tape {
        while let Some(&(n, op)) = self.iter.next() {
            self.step(n, op);
        }
        unimplemented!()
    }

    fn step(&mut self, node: NodeIndex, op: Op) {
        let index = self.mapping.len().try_into().unwrap();
        match op {
            Op::Var(v) => {
                let arg = match self.vars.get_by_index(v).unwrap().as_str() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    _ => panic!(),
                };
                self.tape.push(ClauseOp64::Input);
                self.data.push(index);
                self.data.push(arg);
            }
            Op::Const(c) => {
                // Skip this (because it's not inserted into the tape),
                // recording its value for use as an immediate later.
                self.constants.insert(node, c as f32);
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);

                let f = match op {
                    BinaryOpcode::Add => (
                        ClauseOp64::AddRegReg,
                        ClauseOp64::AddRegImm,
                        ClauseOp64::AddRegImm,
                    ),
                    BinaryOpcode::Mul => (
                        ClauseOp64::MulRegReg,
                        ClauseOp64::MulRegImm,
                        ClauseOp64::MulRegImm,
                    ),
                    BinaryOpcode::Sub => (
                        ClauseOp64::SubRegReg,
                        ClauseOp64::SubRegImm,
                        ClauseOp64::SubImmReg,
                    ),
                };

                self.data.push(index);
                match (lhs, rhs) {
                    (Location::Register(lhs), Location::Register(rhs)) => {
                        self.tape.push(f.0);
                        self.data.push(index);
                        self.data.push(lhs);
                        self.data.push(rhs);
                    }
                    (Location::Register(arg), Location::Immediate(imm)) => {
                        self.tape.push(f.1);
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                    }
                    (Location::Immediate(imm), Location::Register(arg)) => {
                        self.tape.push(f.2);
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
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

                let f = match op {
                    BinaryChoiceOpcode::Min => {
                        (ClauseOp64::MinRegReg, ClauseOp64::MinRegImm)
                    }
                    BinaryChoiceOpcode::Max => {
                        (ClauseOp64::MaxRegReg, ClauseOp64::MaxRegImm)
                    }
                };

                match (lhs, rhs) {
                    (Location::Register(lhs), Location::Register(rhs)) => {
                        self.tape.push(f.0);
                        self.data.push(index);
                        self.data.push(lhs);
                        self.data.push(rhs);
                    }
                    (Location::Register(arg), Location::Immediate(imm)) => {
                        self.tape.push(f.1);
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                    }
                    (Location::Immediate(imm), Location::Register(arg)) => {
                        self.tape.push(f.1);
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
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
                let op = match op {
                    UnaryOpcode::Neg => ClauseOp64::NegReg,
                    UnaryOpcode::Abs => ClauseOp64::AbsReg,
                    UnaryOpcode::Recip => ClauseOp64::RecipReg,
                    UnaryOpcode::Sqrt => ClauseOp64::SqrtReg,
                    UnaryOpcode::Square => ClauseOp64::SquareReg,
                };
                self.tape.push(op);
                self.data.push(index);
                self.data.push(lhs);
            }
        };
        let r = self.mapping.insert(node, index);
        assert!(r.is_none());
    }
}
