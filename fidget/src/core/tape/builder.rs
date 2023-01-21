use crate::{
    context::{BinaryOpcode, Context, Node, Op, VarNode},
    eval::tape::Data,
    tape::{alloc::RegisterAllocator, Op as TapeOp},
};

use std::collections::{btree_map::Entry, BTreeMap};

pub(crate) struct Builder {
    tape: Vec<TapeOp<u32>>,

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

    pub fn finish(self, reg_limit: u8) -> Data {
        let mut alloc =
            RegisterAllocator::new(reg_limit, self.tape.len().max(1));
        for &op in self.tape.iter() {
            match op {
                TapeOp::Input(out, arg) => alloc.op_input(out, arg),
                TapeOp::Var(out, var) => alloc.op_var(out, var),
                TapeOp::Reg(op, out, arg) => alloc.op_reg(out, arg, op),
                TapeOp::RegReg(op, out, lhs, rhs) => {
                    alloc.op_reg_reg(out, lhs, rhs, op)
                }
                TapeOp::RegImm(op, out, arg, imm) => {
                    alloc.op_reg_imm(out, arg, imm, op)
                }
                TapeOp::ImmReg(op, out, arg, imm) => {
                    alloc.op_imm_reg(out, arg, imm, op)
                }
                TapeOp::CopyImm(out, imm) => alloc.op_copy_imm(out, imm),
                TapeOp::Store(..) | TapeOp::Load(..) => {
                    panic!("Invalid operation in SSA tape");
                }
            }
        }
        let mut asm = alloc.finalize();
        asm.choice_count = self.choice_count;

        Data::new(asm, self.var_names)
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
            Op::Input(v) => {
                let arg = match ctx.get_var_by_index(v).unwrap() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    i => panic!("Unexpected input index: {i}"),
                };
                Some(TapeOp::Input(index.unwrap(), arg))
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
                Some(TapeOp::Var(index.unwrap(), arg))
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

                if matches!(op, BinaryOpcode::Min | BinaryOpcode::Max) {
                    self.choice_count += 1;
                }

                let op = match (lhs, rhs) {
                    (Location::Slot(lhs), Location::Slot(rhs)) => {
                        TapeOp::RegReg(op, index, lhs, rhs)
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        TapeOp::RegImm(op, index, arg, imm)
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        TapeOp::ImmReg(op, index, arg, imm)
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
                Some(TapeOp::Reg(op, index, lhs))
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
