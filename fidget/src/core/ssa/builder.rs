use crate::{
    context::{BinaryOpcode, Context, Node, Op, UnaryOpcode},
    ssa::{Op as SsaOp, Tape},
};

use std::collections::BTreeMap;

pub struct Builder {
    tape: Vec<SsaOp>,
    data: Vec<u32>,

    mapping: BTreeMap<Node, u32>,
    constants: BTreeMap<Node, f32>,
    choice_count: usize,
}

#[derive(Debug)]
enum Location {
    Slot(u32),
    Immediate(f32),
}

impl Builder {
    pub fn new() -> Self {
        Self {
            tape: vec![],
            data: vec![],
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
        }
    }

    pub fn finish(self) -> Tape {
        Tape {
            tape: self.tape,
            data: self.data,
            choice_count: self.choice_count,
        }
    }

    fn get_allocated_value(&mut self, node: Node) -> Location {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Location::Slot(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Location::Immediate(*c)
        }
    }

    /// Ensure that the given node is mapped.
    ///
    /// This must be called before `step` uses the node (as either parent or
    /// child).
    pub fn declare_node(&mut self, node: Node, op: Op) {
        match op {
            Op::Const(c) => {
                self.constants.insert(node, c.0 as f32);
            }
            _ => {
                let index: u32 = self.mapping.len().try_into().unwrap();
                self.mapping.entry(node).or_insert(index);
            }
        }
    }

    pub fn step(&mut self, node: Node, op: Op, ctx: &Context) {
        let index = self.mapping.get(&node).cloned();
        let op = match op {
            Op::Var(v) => {
                let arg = match ctx.get_var_by_index(v).unwrap() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    i => panic!("Unexpected input index: {i}"),
                };
                self.data.push(index.unwrap());
                self.data.push(arg);
                Some(SsaOp::Input)
            }
            Op::Const(c) => {
                // Skip this (because it's not inserted into the tape),
                // recording its value for use as an immediate later.
                self.constants.insert(node, c.0 as f32);
                assert!(index.is_none());
                None
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);
                let index = index.unwrap();

                let f = match op {
                    BinaryOpcode::Add => {
                        (SsaOp::AddRegReg, SsaOp::AddRegImm, SsaOp::AddRegImm)
                    }
                    BinaryOpcode::Sub => {
                        (SsaOp::SubRegReg, SsaOp::SubRegImm, SsaOp::SubImmReg)
                    }
                    BinaryOpcode::Mul => {
                        (SsaOp::MulRegReg, SsaOp::MulRegImm, SsaOp::MulRegImm)
                    }
                    BinaryOpcode::Div => {
                        (SsaOp::DivRegReg, SsaOp::DivRegImm, SsaOp::DivImmReg)
                    }
                    BinaryOpcode::Min => {
                        (SsaOp::MinRegReg, SsaOp::MinRegImm, SsaOp::MinRegImm)
                    }
                    BinaryOpcode::Max => {
                        (SsaOp::MaxRegReg, SsaOp::MaxRegImm, SsaOp::MaxRegImm)
                    }
                };

                if matches!(op, BinaryOpcode::Min | BinaryOpcode::Max) {
                    self.choice_count += 1;
                }

                let op = match (lhs, rhs) {
                    (Location::Slot(lhs), Location::Slot(rhs)) => {
                        self.data.push(index);
                        self.data.push(lhs);
                        self.data.push(rhs);
                        f.0
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                        f.1
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                        f.2
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                };
                Some(op)
            }
            Op::Unary(op, lhs) => {
                let lhs = match self.get_allocated_value(lhs) {
                    Location::Slot(r) => r,
                    Location::Immediate(..) => {
                        panic!("Cannot handle f(imm)")
                    }
                };
                let index = index.unwrap();
                let op = match op {
                    UnaryOpcode::Neg => SsaOp::NegReg,
                    UnaryOpcode::Abs => SsaOp::AbsReg,
                    UnaryOpcode::Recip => SsaOp::RecipReg,
                    UnaryOpcode::Sqrt => SsaOp::SqrtReg,
                    UnaryOpcode::Square => SsaOp::SquareReg,
                };
                self.data.push(index);
                self.data.push(lhs);
                Some(op)
            }
        };

        if let Some(op) = op {
            self.tape.push(op);
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
