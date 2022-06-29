use std::collections::BTreeMap;

use crate::compiler::Asm;
use crate::indexed::{define_index, IndexVec};

define_index!(VarIndex, "Index of a variable in `Tape::var_data`");
define_index!(RegIndex, "Index of a register");
define_index!(ChoiceIndex, "Index of a min/max choice");

#[derive(Debug)]
enum Instruction {
    Var(VarIndex),
    Const(f64),
    Add(RegIndex, RegIndex),
    Mul(RegIndex, RegIndex),
    Min(RegIndex, RegIndex, ChoiceIndex),
    Max(RegIndex, RegIndex, ChoiceIndex),
    Neg(RegIndex),
    Abs(RegIndex),
    Recip(RegIndex),
    Sqrt(RegIndex),
    Sin(RegIndex),
    Cos(RegIndex),
    Tan(RegIndex),
    Asin(RegIndex),
    Acos(RegIndex),
    Atan(RegIndex),
    Exp(RegIndex),
    Ln(RegIndex),

    /// If any of the choices match, then execute the given set of instructions
    Cond(Vec<(ChoiceIndex, Choice)>, Vec<Instruction>),
}

/// Working data when performing evaluation
#[derive(Debug)]
struct TapeData<T> {
    vars: IndexVec<T, VarIndex>,
    regs: IndexVec<T, RegIndex>,
}

#[derive(Debug)]
struct Tape {
    tape: Vec<Instruction>,

    /// Working data for recording min/max choices (during interval evaluation)
    /// and checking them in conditionals (during float evaluation)
    choices: IndexVec<Choice, ChoiceIndex>,

    /// Working data for floating-point evaluation
    f_data: TapeData<f64>,

    /// Working data for interval evaluation
    i_data: TapeData<(f64, f64)>,

    /// Map of variable names to indexes in `TapeData::vars`
    vars: BTreeMap<String, VarIndex>,
}

/// Represents a choice by a `min` or `max` node.
#[derive(Copy, Clone, Debug)]
enum Choice {
    Left,
    Right,
    Both,
}

impl Tape {
    pub fn eval_f(&self, x: f64, y: f64, z: f64, choices: &[Choice]) -> f64 {
        todo!()
    }
}
