use std::collections::{BTreeMap, BTreeSet};

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
        out.into_iter().flatten()
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
pub struct Block {
    tape: Vec<Instruction>,

    /// Registers which are sourced externally to this block and unmodified
    inputs: BTreeSet<RegIndex>,
    /// Registers which are sourced externally to this block and modified
    outputs: BTreeSet<RegIndex>,
    /// Registers which are only used within this block
    locals: BTreeSet<RegIndex>,
}

impl Block {
    /// Returns an inner block, with `inputs` / `outputs` / `locals`
    /// uninitialized.
    ///
    /// This should be stored within a higher-level tape that is passed into
    /// `Block::new` for finalization.
    pub fn inner(tape: Vec<Instruction>) -> Self {
        Self {
            tape,
            inputs: Default::default(),
            outputs: Default::default(),
            locals: Default::default(),
        }
    }
    /// Builds a top-level `Block` from the given instruction tape.
    fn new(tape: Vec<Instruction>) -> Self {
        let mut out = Self {
            tape,
            inputs: Default::default(),
            outputs: Default::default(),
            locals: Default::default(),
        };
        // Find the root blocks of every register
        let mut reg_blocks = BTreeMap::new();
        out.reg_blocks(&mut vec![], &mut reg_blocks);

        // Turn that data inside out and store the registers at each root block
        let mut reg_paths: BTreeMap<Vec<usize>, Vec<RegIndex>> =
            BTreeMap::new();
        for (r, b) in reg_blocks.iter() {
            reg_paths.entry(b.to_vec()).or_default().push(*r);
        }

        // Recurse down the tree, saving local registers for each block
        out.populate_locals(&mut vec![], &reg_paths);

        // Store input and output registers at each block
        out.populate_io();

        // The root tape should have all local variables
        assert!(out.inputs.is_empty());
        assert!(out.outputs.is_empty());

        out
    }

    /// Populates `self.locals` recursively from the root of the tree.
    fn populate_locals(
        &mut self,
        path: &mut Vec<usize>,
        reg_paths: &BTreeMap<Vec<usize>, Vec<RegIndex>>,
    ) {
        self.locals
            .extend(reg_paths.get(path).into_iter().flat_map(|i| i.iter()));
        for (index, instruction) in self.tape.iter_mut().enumerate() {
            if let Instruction::Cond(_, b) = instruction {
                path.push(index);
                b.populate_locals(path, reg_paths);
                path.pop();
            }
        }
    }

    /// Populates `self.inputs`, `self.outputs`, and `self.locals`
    fn populate_io(&mut self) {
        for instruction in self.tape.iter_mut() {
            // Store registers
            for (r, out) in instruction
                .iter_regs()
                .zip(std::iter::once(true).chain(std::iter::repeat(false)))
            {
                if self.locals.contains(&r) {
                    // Nothing to do here
                } else if out {
                    self.outputs.insert(r);
                } else {
                    self.inputs.insert(r);
                }
            }
            // Recurse into condition blocks
            if let Instruction::Cond(_, b) = instruction {
                b.populate_io();

                // We forward IO from children blocks, excluding registers
                // which are local to this block.
                self.inputs.extend(b.inputs.difference(&self.locals));
                self.outputs.extend(b.outputs.difference(&self.locals));
            }
        }
        // Remove outputs from input set. It's possible for one register
        // to be defined in both sets, if it's assigned in this block then
        // used in both this block and outside of this block.
        self.inputs = self.inputs.difference(&self.outputs).cloned().collect();

        assert!(self.inputs.intersection(&self.locals).next().is_none());
        assert!(self.outputs.intersection(&self.locals).next().is_none());
        assert!(self.inputs.intersection(&self.outputs).next().is_none());
    }

    /// Calculates the block at which each register must be defined.
    ///
    /// Blocks are given as addresses from the root block.
    fn reg_blocks(
        &self,
        path: &mut Vec<usize>,
        out: &mut BTreeMap<RegIndex, Vec<usize>>,
    ) {
        use std::collections::btree_map::Entry;
        for (index, instruction) in self.tape.iter().enumerate() {
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

/// Represents configuration for a generated program
#[derive(Debug)]
pub struct Config {
    /// Number of registers used during evaluation
    pub reg_count: usize,

    /// Number of choice slots used during evaluation
    pub choice_count: usize,
    /// Number of variables needed for evaluation
    pub var_count: usize,

    /// Map of variable names to indexes (in the range `0..var_count`)
    pub vars: BTreeMap<String, VarIndex>,
}

/// Represents a program that can be evaluated or converted to a new form.
///
/// Note that such a block is divorced from the generating `Context`, and
/// can be processed independantly.
///
/// The program is in SSA form, i.e. each register is only assigned to once.
#[derive(Debug)]
pub struct Program {
    tape: Block,
    root: RegIndex,
    config: Config,
}

/// A generated function.
///
/// The opening of the function is omitted but can be reconstructed
/// from the index, inputs, and outputs.
struct MetalFunction {
    index: usize,
    body: String,
    root: bool,
    /// Registers which are sourced externally to this block and unmodified
    inputs: BTreeSet<RegIndex>,
    /// Registers which are sourced externally to this block and modified
    outputs: BTreeSet<RegIndex>,
}

impl MetalFunction {
    fn declaration(&self) -> String {
        let mut out = String::new();
        out += &formatdoc!(
            "
            inline {} t_shape_{}(
                const device float* vars, const device uint8_t* choices",
            if self.root { "float" } else { "void" },
            self.index
        );
        let mut first = true;
        for i in &self.inputs {
            if first {
                out += ",\n    ";
            } else {
                out += ", ";
            }
            first = false;
            out += &format!("const float v{}", i.0);
        }
        let mut first = true;
        for i in &self.outputs {
            if first {
                out += ",\n    ";
            } else {
                out += ", ";
            }
            first = false;
            out += &format!("thread float& v{}", i.0);
        }
        out += "\n)";
        out
    }
    /// Generates text to call a function
    fn call(&self) -> String {
        let mut out = String::new();
        out += &format!("t_shape_{}(vars, choices", self.index);

        for i in &self.inputs {
            out += &format!(", v{}", i.0);
        }
        for i in &self.outputs {
            out += &format!(", v{}", i.0);
        }
        out += ");";
        out
    }
}

impl Program {
    pub fn from_compiler(c: &Compiler) -> Self {
        let mut regs = IndexMap::default();
        let mut vars = IndexMap::default();
        let mut choices = IndexMap::default();
        let tape = Block::new(c.to_tape(&mut regs, &mut vars, &mut choices));

        let var_names = vars
            .iter()
            .map(|(vn, vi)| {
                (c.ctx.get_var_by_index(*vn).unwrap().to_string(), *vi)
            })
            .collect();

        Self {
            tape,
            config: Config {
                reg_count: regs.len(),
                var_count: vars.len(),
                choice_count: choices.len(),
                vars: var_names,
            },
            root: regs.insert(c.root),
        }
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Converts the program to a Metal shader
    pub fn to_metal(&self) -> String {
        let mut out = formatdoc!(
            "
            #define VAR_COUNT {}
            #define CHOICE_COUNT {}

            {}
            ",
            self.config.var_count,
            self.config.choice_count,
            METAL_PRELUDE_FLOAT
        );

        // Global map from block paths to (function index, body)
        let mut functions: BTreeMap<Vec<usize>, MetalFunction> =
            BTreeMap::new();
        self.to_metal_inner(&self.tape, &mut vec![], &mut functions);

        out += "// Forward declarations\n";
        for f in functions.values() {
            out += &format!("{};\n", f.declaration());
        }
        out += "\n// Function definitions\n";
        for f in functions.values() {
            out += &format!("{} {{\n{}}}\n", f.declaration(), f.body);
        }
        out += "\n";
        out += &formatdoc!(
            "
        // Root function
        inline float t_eval(const device float* vars,
                            const device uint8_t* choices)
        {{
            return t_shape_{}(vars, choices);
        }}",
            functions.get(&vec![]).unwrap().index
        );
        out += METAL_KERNEL_FLOAT;
        out
    }

    fn to_metal_inner(
        &self,
        block: &Block,
        path: &mut Vec<usize>,
        functions: &mut BTreeMap<Vec<usize>, MetalFunction>,
    ) {
        let mut first = true;
        let mut out = String::new();
        for r in block.locals.iter() {
            if first {
                out += "    float ";
                first = false;
            } else {
                out += ", ";
            }
            out += &format!("v{}", r.0);
        }
        if !first {
            out += ";\n"
        }
        for (index, instruction) in block.tape.iter().enumerate() {
            if let Some(i) = instruction.to_metal() {
                for line in i.lines() {
                    out += &format!("    {}\n", line);
                }
            } else if let Instruction::Cond(cond, next) = &instruction {
                // Recurse!
                path.push(index);
                self.to_metal_inner(next, path, functions);
                let f = functions.get(path).unwrap();
                path.pop();

                // Write out the conditional, calling the inner function
                out += "    if (";
                if cond.len() > 1 {
                    let mut first = true;
                    for c in cond {
                        if first {
                            first = false;
                        } else {
                            out += " || ";
                        }
                        out += &format!(
                            "(choices[{}] & {})",
                            c.0 .0,
                            c.1.to_metal()
                        );
                    }
                } else {
                    out += &format!(
                        "choices[{}] & {}",
                        cond[0].0 .0,
                        cond[0].1.to_metal()
                    );
                }
                out += ") {\n        ";
                out += &f.call();
                out += "\n    }\n";
            } else {
                panic!("Could not get out register or Cond block");
            }
        }
        let i = functions.len();
        let root = path.is_empty();
        if root {
            out += &format!("    return v{};\n", self.root.0);
        }
        functions.insert(
            path.clone(),
            MetalFunction {
                index: i,
                body: out,
                root,
                inputs: block.inputs.clone(),
                outputs: block.outputs.clone(),
            },
        );
    }
}

const METAL_PRELUDE_FLOAT: &str = r#"
#include <metal_stdlib>

#define RHS 1
#define LHS 2

// Shapes
inline float t_mul(float a, float b) {
    return a * b;
}
inline float t_add(float a, float b) {
    return a + b;
}
inline float t_min(float a, float b) {
    return metal::fmin(a, b);
}
inline float t_max(float a, float b) {
    return metal::fmax(a, b);
}
inline float t_neg(float a) {
    return -a;
}
inline float t_sqrt(float a) {
    return metal::sqrt(a);
}
inline float t_const(float a) {
    return a;
}
"#;

const METAL_KERNEL_FLOAT: &str = r#"
kernel void main0(const device float* vars [[buffer(0)]],
                  const device uint8_t* choices [[buffer(1)]],
                  device float* result [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    result[index] = t_eval(&vars[index * VAR_COUNT],
                           &choices[index * CHOICE_COUNT]);
}
"#;
