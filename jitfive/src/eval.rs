use std::collections::BTreeMap;

use crate::indexed::IndexVec;
use crate::program::{Choice, ChoiceIndex, Program, RegIndex, VarIndex};

/// Working data when performing evaluation
#[derive(Debug)]
struct TapeData<T> {
    vars: IndexVec<T, VarIndex>,
    regs: IndexVec<T, RegIndex>,
}

#[derive(Debug)]
struct Tape<'a> {
    prog: &'a Program,

    /// Working data for recording min/max choices (during interval evaluation)
    /// and checking them in conditionals (during float evaluation)
    choices: IndexVec<Choice, ChoiceIndex>,

    /// Working data for floating-point evaluation
    f_data: TapeData<f64>,

    /// Working data for interval evaluation
    i_data: TapeData<(f64, f64)>,
}

impl Tape<'_> {
    pub fn eval_f(&self, x: f64, y: f64, z: f64, choices: &[Choice]) -> f64 {
        todo!()
    }
}
