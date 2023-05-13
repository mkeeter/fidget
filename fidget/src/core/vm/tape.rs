use std::{collections::BTreeMap, sync::Arc};

use crate::{
    eval::{Choice, Family},
    vm::Op,
};

/// Evaluation tape, which is an ordered set of functions
///
/// It is parameterized by an [`Family`](Family) type, which sets the register
/// count of the inner VM tapes.
///
/// This is a heavy-weight value and should not change once constructed;
/// consider wrapping it in an `Arc` for shared light-weight use.
pub struct Tape<F> {
    /// The tape groups are stored in reverse order, such that the root of the
    /// tree is the first item in the first tape
    pub data: Vec<ChoiceTape>,

    /// Number of choice operations
    choice_count: usize,

    /// Number of slots used by the tape
    slot_count: usize,

    /// Mapping from variable names (in the original
    /// [`Context`](crate::context::Context)) to indexes in the variable array
    /// used during evaluation.
    vars: BTreeMap<String, u32>,

    _p: std::marker::PhantomData<fn() -> F>,
}

impl<F> Tape<F> {
    /// Returns the number of slots used by this tape
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
    /// Returns the number of variables used by this tape
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }
    /// Returns the mapping from variable name to slot
    pub fn vars(&self) -> &BTreeMap<String, u32> {
        &self.vars
    }
    /// Returns the number of choices written by this tape
    pub fn choice_count(&self) -> usize {
        self.choice_count
    }

    /// Produces an iterator that visits [`vm::Op`](crate::vm::Op) values in
    /// evaluation order.
    pub fn iter_asm<'a>(
        &'a self,
        active_groups: &'a [usize],
    ) -> impl Iterator<Item = Op> + 'a {
        active_groups
            .iter()
            .rev()
            .flat_map(|i| self.data[*i].tape.iter().rev())
            .cloned()
    }
}

impl<F: Family> Tape<F> {
    /// Returns the register limit used when planning this tape
    pub fn reg_limit(&self) -> u8 {
        F::REG_LIMIT
    }
}

/// A tape alongside the choices which lead it to being selected
pub struct ChoiceTape {
    /// List of instructions in reverse-evaluation order
    pub tape: Vec<Op>,

    /// Array of `(choice index, choice)` tuples
    ///
    /// If any of the tuples matches, then this tape is active
    ///
    /// As a special case, the always-selected (root) tape is represented with
    /// an empty vector.
    pub choices: Vec<(usize, Choice)>,
}

impl<F: Family> Tape<F> {
    /// Builds a new tape, built from many operation groups
    pub fn new(data: Vec<ChoiceTape>, vars: BTreeMap<String, u32>) -> Self {
        let choice_count = data
            .iter()
            .flat_map(|t| t.choices.iter())
            .map(|c| c.0 + 1)
            .max()
            .unwrap_or(0) as usize;
        let slot_count = data
            .iter()
            .flat_map(|t| t.tape.iter())
            .flat_map(|op| op.iter_slots())
            .map(|s| s + 1)
            .max()
            .unwrap_or(0) as usize;
        Self {
            data,
            vars,
            choice_count,
            slot_count,
            _p: std::marker::PhantomData,
        }
    }
}

/// Represents a tape specialized with a set of choices and active groups
///
/// This is a handle expected to be passed by value
pub struct SpecializedTape<F> {
    /// Root tape, which contains all groups
    pub tape: Arc<Tape<F>>,

    /// Set of choices, indicating which `min` and `max` clauses are specialized
    pub choices: Vec<Choice>,

    /// Currently active groups, as indexes into the root tape
    active_groups: Vec<usize>,
}

impl<F> SpecializedTape<F> {
    /// Iterates over active clauses in the tape in evaluation order
    pub fn iter_asm<'a>(&'a self) -> impl Iterator<Item = Op> + 'a {
        self.tape.iter_asm(&self.active_groups)
    }
}
