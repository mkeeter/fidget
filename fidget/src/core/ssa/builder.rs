use crate::{
    context::{BinaryOpcode, Context, Node, Op, UnaryOpcode, VarNode},
    ssa::{Op as SsaOp, Tape},
};

use std::{
    collections::{btree_map::Entry, BTreeMap},
    sync::Arc,
};

pub(crate) struct Builder {
    tape: Vec<SsaOp>,

    mapping: BTreeMap<Node, u32>,
    vars: BTreeMap<VarNode, u32>,
    var_names: BTreeMap<String, u32>,
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
            mapping: BTreeMap::new(),
            vars: BTreeMap::new(),
            var_names: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
        }
    }

    pub fn finish(self) -> Tape {
        Tape {
            tape: self.tape,
            choice_count: self.choice_count,
            vars: Arc::new(self.var_names),
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
    ///
    /// Returns the SSA index assigned to this node.
    pub fn declare_node(&mut self, node: Node, op: Op) -> Option<u32> {
        match op {
            Op::Const(c) => {
                self.constants.insert(node, c.0 as f32);
                None
            }
            _ => {
                let index: u32 = self.mapping.len().try_into().unwrap();
                Some(*self.mapping.entry(node).or_insert(index))
            }
        }
    }

    pub fn step(&mut self, node: Node, op: Op, ctx: &Context) {
        let index = self.mapping.get(&node).cloned();
        let op = match op {
            Op::Input(v) => {
                let arg = match ctx.get_var_by_index(v).unwrap() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    i => panic!("Unexpected input index: {i}"),
                };
                Some(SsaOp::Input(index.unwrap(), arg))
            }
            Op::Var(v) => {
                let next_var = self.vars.len().try_into().unwrap();
                let arg = match self.vars.entry(v) {
                    Entry::Vacant(e) => {
                        e.insert(next_var);
                        let name = ctx.get_var_by_index(v).unwrap().to_owned();
                        self.var_names.insert(name, next_var);
                        next_var
                    }
                    Entry::Occupied(a) => *a.get(),
                };
                Some(SsaOp::Var(index.unwrap(), arg))
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

                type RegFn = fn(u32, u32, u32) -> SsaOp;
                type ImmFn = fn(u32, u32, f32) -> SsaOp;
                let f: (RegFn, ImmFn, ImmFn) = match op {
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
                        f.0(index, lhs, rhs)
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        f.1(index, arg, imm)
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        f.2(index, arg, imm)
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
                Some(op(index, lhs))
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
