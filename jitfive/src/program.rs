use std::collections::{BTreeMap, BTreeSet};

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
    pub fn out_reg(&self) -> Option<RegIndex> {
        self.iter_regs().next()
    }
    pub fn name(&self) -> &str {
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
    pub tape: Vec<Instruction>,

    /// Does the block or any inner block contain variable lookups?
    pub has_var: bool,

    /// Does the block or any inner block contain choice lookups?
    pub has_choice: bool,

    /// Number of instructions (including inner block)
    pub weight: usize,

    /// Registers which are sourced externally to this block and unmodified
    pub inputs: BTreeSet<RegIndex>,
    /// Registers which are sourced externally to this block and modified
    pub outputs: BTreeSet<RegIndex>,
    /// Registers which are only used within this block
    pub locals: BTreeSet<RegIndex>,
}

impl Block {
    /// Returns an inner block, with `inputs` / `outputs` / `locals`
    /// uninitialized.
    ///
    /// This should be stored within a higher-level tape that is passed into
    /// `Block::new` for finalization.
    pub fn inner(tape: Vec<Instruction>) -> Self {
        let has_var = tape.iter().any(|i| match i {
            Instruction::Var { .. } => true,
            Instruction::Cond(_, b) => b.has_var,
            _ => false,
        });
        let has_choice = tape.iter().any(|i| match i {
            Instruction::Cond(_, b) => b.has_choice,
            Instruction::Min { .. } | Instruction::Max { .. } => true,
            _ => false,
        });
        let weight = tape
            .iter()
            .map(|i| match i {
                Instruction::Cond(_, b) => b.weight,
                _ => 1,
            })
            .sum::<usize>();
        Self {
            tape,
            has_choice,
            has_var,
            weight,
            inputs: Default::default(),
            outputs: Default::default(),
            locals: Default::default(),
        }
    }
    /// Builds a top-level `Block` from the given instruction tape.
    fn new(tape: Vec<Instruction>) -> Self {
        let mut out = Self::inner(tape);

        // Find the root blocks of every register
        let mut reg_blocks = BTreeMap::new();
        out.reg_blocks(&mut vec![], &mut reg_blocks);

        // Turn that data inside out and store the registers at each root block
        let mut reg_paths: BTreeMap<Vec<usize>, BTreeSet<RegIndex>> =
            BTreeMap::new();
        for (r, b) in reg_blocks.iter() {
            reg_paths.entry(b.to_vec()).or_default().insert(*r);
        }

        // Recurse down the tree, saving local registers for each block
        out.populate_locals(&mut vec![], &mut reg_paths);

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
        reg_paths: &mut BTreeMap<Vec<usize>, BTreeSet<RegIndex>>,
    ) {
        if let Some(v) = reg_paths.remove(path) {
            assert!(self.locals.is_empty());
            self.locals = v;
        }
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
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum Choice {
    Left,
    Right,
    Both,
}

/// Represents configuration for a generated program
#[derive(Debug, Clone)]
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
    pub tape: Block,
    pub root: RegIndex,
    config: Config,
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
                (c.ctx.get_var_by_index(*vn).unwrap().to_string(), vi)
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
}
