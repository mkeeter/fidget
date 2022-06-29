use std::collections::BTreeMap;

use crate::{
    compiler::Compiler,
    indexed::{define_index, IndexMap},
};

define_index!(VarIndex, "Index of a variable in a `Program`");
define_index!(RegIndex, "Index of a register in a `Program`");
define_index!(ChoiceIndex, "Index of a min/max choice in a `Program`");

#[derive(Debug)]
pub enum Instruction {
    Var {
        var: VarIndex,
        out: RegIndex,
    },
    Const {
        value: f64,
        out: RegIndex,
    },
    Add {
        lhs: RegIndex,
        rhs: RegIndex,
        out: RegIndex,
    },
    Mul {
        lhs: RegIndex,
        rhs: RegIndex,
        out: RegIndex,
    },
    Min {
        choice: ChoiceIndex,
        lhs: RegIndex,
        rhs: RegIndex,
        out: RegIndex,
    },
    Max {
        choice: ChoiceIndex,
        lhs: RegIndex,
        rhs: RegIndex,
        out: RegIndex,
    },
    Neg {
        reg: RegIndex,
        out: RegIndex,
    },
    Abs {
        reg: RegIndex,
        out: RegIndex,
    },
    Recip {
        reg: RegIndex,
        out: RegIndex,
    },
    Sqrt {
        reg: RegIndex,
        out: RegIndex,
    },
    Sin {
        reg: RegIndex,
        out: RegIndex,
    },
    Cos {
        reg: RegIndex,
        out: RegIndex,
    },
    Tan {
        reg: RegIndex,
        out: RegIndex,
    },
    Asin {
        reg: RegIndex,
        out: RegIndex,
    },
    Acos {
        reg: RegIndex,
        out: RegIndex,
    },
    Atan {
        reg: RegIndex,
        out: RegIndex,
    },
    Exp {
        reg: RegIndex,
        out: RegIndex,
    },
    Ln {
        reg: RegIndex,
        out: RegIndex,
    },

    /// If any of the choices match, then execute the given set of instructions
    Cond(Vec<(ChoiceIndex, Choice)>, Vec<Instruction>),
}

/// Represents a choice by a `min` or `max` node.
#[derive(Copy, Clone, Debug)]
pub enum Choice {
    Left,
    Right,
    Both,
}

/// Represents a program that can be evaluated or converted to a new form.
///
/// Note that such a block is divorced from the generating `Context`, and
/// can be processed independantly.
#[derive(Debug)]
pub struct Program {
    tape: Vec<Instruction>,

    /// Number of registers used during evaluation
    reg_count: usize,
    /// Number of choice slots used during evaluation
    choice_count: usize,
    /// Number of variables needed for evaluation
    var_count: usize,

    /// Map of variable names to indexes (in the range `0..var_count`)
    vars: BTreeMap<String, VarIndex>,
}

impl Program {
    pub fn from_compiler(c: &Compiler) -> Self {
        let mut regs = IndexMap::default();
        let mut vars = IndexMap::default();
        let mut choices = IndexMap::default();
        let tape = c.to_tape(&mut regs, &mut vars, &mut choices);
        Self {
            tape,
            reg_count: regs.len(),
            var_count: vars.len(),
            choice_count: choices.len(),
            vars: Default::default(), // TODO
        }
    }
}

/*
impl Asm {
    pub fn node_count(&self) -> usize {
        match self {
            Asm::Eval(..) | Asm::Var(..) => 1,
            Asm::Cond(_, asm) => asm.iter().map(Asm::node_count).sum(),
        }
    }

    pub fn pretty_print(&self) {
        self.pprint_inner(0)
    }
    fn pprint_inner(&self, indent: usize) {
        match self {
            Asm::Eval(node, op @ Op::Min(a, b))
            | Asm::Eval(node, op @ Op::Max(a, b)) => {
                println!("{:indent$}Match({:?},", "", node);
                println!("{:indent$}  Left => {:?},", "", a);
                println!("{:indent$}  Right => {:?},", "", b);
                println!("{:indent$}  _ => {:?}", "", op);
                println!("{:indent$})", "");
            }
            Asm::Eval(..) | Asm::Var(..) => {
                println!("{:indent$}{:?}", "", self)
            }
            Asm::Cond(src, asm) => {
                println!("{:indent$}Cond(Or({:?}),", "", src);
                for v in asm {
                    v.pprint_inner(indent + 2);
                }
                println!("{:indent$})", "");
            }
        }
    }
}
*/
