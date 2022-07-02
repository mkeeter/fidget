use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use indoc::{formatdoc, writedoc};

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
    /// Returns an interator over registers used in this instruction (both as
    /// inputs and outputs).
    ///
    /// The iterator is guaranteed to put the output register first.
    ///
    /// Returns nothing for [Instruction::Cond].
    ///
    /// ```
    /// # use jitfive::program::{Instruction, RegIndex};
    /// let i = Instruction::Neg { reg: 0.into(), out: 1.into() };
    /// let mut iter = i.iter_regs();
    /// assert_eq!(iter.next(), Some(1.into()));
    /// assert_eq!(iter.next(), Some(0.into()));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_regs(&self) -> impl Iterator<Item = RegIndex> {
        let out = match self {
            Self::Var { out, .. } | Self::Const { out, .. } => {
                [Some(*out), None, None]
            }
            Self::Add { lhs, rhs, out }
            | Self::Mul { lhs, rhs, out }
            | Self::Min { lhs, rhs, out, .. }
            | Self::Max { lhs, rhs, out, .. } => {
                [Some(*out), Some(*lhs), Some(*rhs)]
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
            | Self::Ln { reg, out } => [Some(*out), Some(*reg), None],
            Self::Cond(..) => [None, None, None],
        };
        out.into_iter().filter_map(|i| i)
    }
    fn to_metal(&self) -> Option<String> {
        let out = self.out_reg()?.0;
        Some(match self {
            Self::Var { var, .. } => {
                format!("v{} = t_const(vars[{}]);", out, var.0)
            }
            Self::Const { value, .. } => {
                format!("v{} = t_const({});", out, value)
            }
            Self::Mul { lhs, rhs, .. } | Self::Add { lhs, rhs, .. } => {
                format!("v{} = t_{}(v{}, v{});", out, self.name(), lhs.0, rhs.0)
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
                        default: v{1} = t_{4}(v{2}, v{3}); break;
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
                format!("v{} = t_{}(v{});", out, self.name(), reg.0)
            }
            Self::Cond(..) => return None,
        })
    }
    fn out_reg(&self) -> Option<RegIndex> {
        self.iter_regs().next()
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
    fn max_depth(&self) -> usize {
        1 + self
            .0
            .iter()
            .filter_map(|i| {
                if let Instruction::Cond(_, b) = i {
                    Some(b.max_depth())
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0)
    }
    /// Calculates the block at which each register must be defined
    fn reg_blocks(
        &self,
        path: &mut Vec<usize>,
        out: &mut BTreeMap<RegIndex, Vec<usize>>,
    ) {
        use std::collections::btree_map::Entry;
        for (index, instruction) in self.0.iter().enumerate() {
            for r in instruction.iter_regs() {
                match out.entry(r) {
                    Entry::Vacant(v) => {
                        v.insert(path.clone());
                    }
                    Entry::Occupied(mut v) => {
                        // Find the longest common prefix, which is the block
                        // in which the register should be defined.
                        let prefix_len = v
                            .get()
                            .iter()
                            .zip(path.iter())
                            .take_while(|(a, b)| a == b)
                            .count();
                        assert!(prefix_len <= v.get().len());
                        v.get_mut().resize(prefix_len, usize::MAX);
                    }
                }
            }
            if let Instruction::Cond(_, b) = instruction {
                path.push(index);
                b.reg_blocks(path, out);
                path.pop();
            }
        }
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
    root: RegIndex,

    /// Number of registers used during evaluation
    reg_count: usize,

    /// Represents registers that must be defined in each block.
    /// The key is a path to the block (i.e. the empty key is the root block).
    reg_paths: BTreeMap<Vec<usize>, Vec<RegIndex>>,

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
        let tape = Block(c.to_tape(&mut regs, &mut vars, &mut choices));

        let var_names = vars
            .iter()
            .map(|(vn, vi)| {
                (c.ctx.get_var_by_index(*vn).unwrap().to_string(), *vi)
            })
            .collect();

        // Find the root blocks of every register
        let mut reg_blocks = BTreeMap::new();
        tape.reg_blocks(&mut vec![], &mut reg_blocks);
        let mut reg_paths: BTreeMap<Vec<usize>, Vec<RegIndex>> =
            BTreeMap::new();
        for (r, b) in reg_blocks.into_iter() {
            reg_paths.entry(b).or_default().push(r);
        }

        Self {
            tape,
            reg_paths,
            reg_count: regs.len(),
            var_count: vars.len(),
            choice_count: choices.len(),
            vars: var_names,
            root: regs.insert(c.root),
        }
    }
    pub fn write_metal<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writedoc!(w, "
            #include <metal_stdlib>

            #define RHS 1
            #define LHS 2

            float t_mul(float a, float b) {{
                return a * b;
            }}
            float t_add(float a, float b) {{
                return a + b;
            }}
            float t_min(float a, float b) {{
                return metal::fmin(a, b);
            }}
            float t_max(float a, float b) {{
                return metal::fmax(a, b);
            }}
            float t_neg(float a) {{
                return -a;
            }}
            float t_sqrt(float a) {{
                return metal::sqrt(a);
            }}
            float t_const(float a) {{
                return a;
            }}
            float hello(const device float* vars, const device uint8_t* choices) {{
        ")?;

        // Disable indentation for large shaders
        let indent = if self.tape.max_depth() > 16 { 0 } else { 4 };
        self.as_metal_inner(w, &self.tape, &mut vec![], indent)?;
        writeln!(w, "    return v{};\n}}", self.root.0)?;
        Ok(())
    }
    fn as_metal_inner<W: Write>(
        &self,
        w: &mut W,
        block: &Block,
        path: &mut Vec<usize>,
        indent: usize,
    ) -> std::io::Result<()> {
        let mut first = true;
        for r in self.reg_paths.get(path).into_iter().flat_map(|i| i.iter()) {
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
        for (index, instruction) in block.0.iter().enumerate() {
            if let Some(out) = instruction.to_metal() {
                for line in out.lines() {
                    writeln!(w, "{:indent$}{}", "", line)?;
                }
            } else if let Instruction::Cond(cond, next) = &instruction {
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
                path.push(index);
                self.as_metal_inner(
                    w,
                    next,
                    path,
                    if indent > 0 { indent + 4 } else { 0 },
                )?;
                path.pop();
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
