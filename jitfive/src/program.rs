use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use indoc::formatdoc;

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
    Cond(Vec<(ChoiceIndex, Choice)>, Block),
}

impl Instruction {
    fn accumulate_regs(&self, set: &mut BTreeSet<RegIndex>) {
        match self {
            Self::Var { out, .. } | Self::Const { out, .. } => {
                set.insert(*out);
            }
            Self::Add { lhs, rhs, out }
            | Self::Mul { lhs, rhs, out }
            | Self::Min { lhs, rhs, out, .. }
            | Self::Max { lhs, rhs, out, .. } => {
                set.insert(*out);
                set.insert(*lhs);
                set.insert(*rhs);
            }
            Self::Neg { reg, out }
            | Self::Abs { reg, out }
            | Self::Recip { reg, out }
            | Self::Sqrt { reg, out }
            | Self::Sin { reg, out }
            | Self::Cos { reg, out }
            | Self::Tan { reg, out }
            | Self::Asin { reg, out }
            | Self::Acos { reg, out }
            | Self::Atan { reg, out }
            | Self::Exp { reg, out }
            | Self::Ln { reg, out } => {
                set.insert(*reg);
                set.insert(*out);
            }
            Self::Cond(..) => (),
        }
    }
    fn to_metal(&self) -> Option<String> {
        let out = self.out_reg()?.0;
        Some(match self {
            Self::Var { var, .. } => format!("v{} = vars[{}];", out, var.0),
            Self::Const { value, .. } => format!("v{} = {};", out, value),
            Self::Mul { lhs, rhs, .. } | Self::Add { lhs, rhs, .. } => {
                format!("v{} = {}(v{}, v{});", out, self.name(), lhs.0, rhs.0)
            }
            Self::Max {
                lhs, rhs, choice, ..
            }
            | Self::Min {
                lhs, rhs, choice, ..
            } => {
                formatdoc!(
                    "switch (choices[{0}]) {{
                        case LHS: v{1} = v{2}; break;
                        case RHS: v{1} = v{3}; break;
                        default: v{1} = {4}(v{2}, v{3}); break;
                    }}",
                    choice.0,
                    out,
                    lhs.0,
                    rhs.0,
                    self.name(),
                )
            }
            Self::Ln { reg, .. }
            | Self::Exp { reg, .. }
            | Self::Atan { reg, .. }
            | Self::Acos { reg, .. }
            | Self::Asin { reg, .. }
            | Self::Tan { reg, .. }
            | Self::Cos { reg, .. }
            | Self::Sin { reg, .. }
            | Self::Sqrt { reg, .. }
            | Self::Recip { reg, .. }
            | Self::Abs { reg, .. }
            | Self::Neg { reg, .. } => {
                format!("v{} = {}(v{});", out, self.name(), reg.0)
            }
            Self::Cond(..) => return None,
        })
    }
    fn out_reg(&self) -> Option<RegIndex> {
        match self {
            Self::Var { out, .. }
            | Self::Const { out, .. }
            | Self::Add { out, .. }
            | Self::Mul { out, .. }
            | Self::Min { out, .. }
            | Self::Max { out, .. }
            | Self::Neg { out, .. }
            | Self::Abs { out, .. }
            | Self::Recip { out, .. }
            | Self::Sqrt { out, .. }
            | Self::Sin { out, .. }
            | Self::Cos { out, .. }
            | Self::Tan { out, .. }
            | Self::Asin { out, .. }
            | Self::Acos { out, .. }
            | Self::Atan { out, .. }
            | Self::Exp { out, .. }
            | Self::Ln { out, .. } => Some(*out),
            Self::Cond(..) => None,
        }
    }
    fn name(&self) -> &str {
        match self {
            Self::Var { .. } => "var",
            Self::Const { .. } => "const",
            Self::Cond(..) => "cond",
            Self::Add { .. } => "add",
            Self::Mul { .. } => "mul",
            Self::Min { .. } => "min",
            Self::Max { .. } => "max",
            Self::Neg { .. } => "neg",
            Self::Abs { .. } => "abs",
            Self::Recip { .. } => "recip",
            Self::Sqrt { .. } => "sqrt",
            Self::Sin { .. } => "sin",
            Self::Cos { .. } => "cos",
            Self::Tan { .. } => "tan",
            Self::Asin { .. } => "asin",
            Self::Acos { .. } => "acos",
            Self::Atan { .. } => "atan",
            Self::Exp { .. } => "exp",
            Self::Ln { .. } => "ln",
        }
    }
}

#[derive(Debug)]
pub struct Block(pub Vec<Instruction>);
impl Block {
    /// Returns a set of registers used in this block of the program,
    /// **without** recursing into `Cond` instructions.
    fn my_regs(&self) -> BTreeSet<RegIndex> {
        let mut out = BTreeSet::new();
        for r in &self.0 {
            r.accumulate_regs(&mut out);
        }
        out
    }
}

/// Represents a choice by a `min` or `max` node.
#[derive(Copy, Clone, Debug)]
pub enum Choice {
    Left,
    Right,
    Both,
}

impl Choice {
    fn to_metal(self) -> &'static str {
        match self {
            Self::Left => "LHS",
            Self::Right => "RHS",
            Self::Both => "BOTH",
        }
    }
}

/// Represents a program that can be evaluated or converted to a new form.
///
/// Note that such a block is divorced from the generating `Context`, and
/// can be processed independantly.
#[derive(Debug)]
pub struct Program {
    tape: Block,

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

        let var_names = vars
            .iter()
            .map(|(vn, vi)| {
                (c.ctx.get_var_by_index(*vn).unwrap().to_string(), *vi)
            })
            .collect();
        Self {
            tape: Block(tape),
            reg_count: regs.len(),
            var_count: vars.len(),
            choice_count: choices.len(),
            vars: var_names,
        }
    }
    pub fn write_metal<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "float hello(const device float* vars, const device Choice* choices) {{")?;
        self.as_metal_inner(w, &self.tape, BTreeSet::default(), 4)?;
        writeln!(w, "}}")?;
        Ok(())
    }
    fn as_metal_inner<W: Write>(
        &self,
        w: &mut W,
        block: &Block,
        parent_regs: BTreeSet<RegIndex>,
        indent: usize,
    ) -> std::io::Result<()> {
        let my_regs = block.my_regs();
        let mut first = true;
        for r in my_regs.difference(&parent_regs) {
            if first {
                write!(w, "{:indent$}float ", "")?;
                first = false;
            } else {
                write!(w, ", ")?;
            }
            write!(w, "v{}", r.0)?;
        }
        if !first {
            writeln!(w, ";")?;
        }
        for b in &block.0 {
            if let Some(out) = b.to_metal() {
                for line in out.lines() {
                    writeln!(w, "{:indent$}{}", "", line)?;
                }
            } else if let Instruction::Cond(cond, next) = &b {
                write!(w, "{:indent$}if (", "")?;
                if cond.len() > 1 {
                    let mut first = true;
                    for c in cond {
                        if first {
                            first = false;
                        } else {
                            write!(w, " || ")?;
                        }
                        write!(
                            w,
                            "(choices[{}] & {})",
                            c.0 .0,
                            c.1.to_metal()
                        )?;
                    }
                } else {
                    write!(
                        w,
                        "choices[{}] & {}",
                        cond[0].0 .0,
                        cond[0].1.to_metal()
                    )?;
                }
                writeln!(w, ") {{")?;
                let new_regs = parent_regs.union(&my_regs).cloned().collect();
                self.as_metal_inner(w, next, new_regs, indent + 4)?;
                writeln!(w, "{:indent$}}}", "")?;
            } else {
                panic!("Could not get out register or Cond block");
            }
        }
        Ok(())
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
