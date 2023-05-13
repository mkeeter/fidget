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
pub struct TapeData<F> {
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

impl<F> TapeData<F> {
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

    /// Returns the total number of operations in the tape
    ///
    /// This is the sum of all operations in the node groups.  Due to inlining
    /// and the insertion of `Load` and `Store` operations, it may be larger
    /// than the raw number of arithmetic operations in the expression!
    pub fn len(&self) -> usize {
        self.data.iter().map(|c| c.tape.len()).sum()
    }
}

impl<F: Family> TapeData<F> {
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

impl<F: Family> TapeData<F> {
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
/// This is a heavy-weight data structure that's typically wrapped in a
/// [`Tape`] for ease of use.
pub struct InnerTape<F> {
    /// Root tape, which contains all groups
    pub tape: Arc<TapeData<F>>,

    /// Set of choices, indicating which `min` and `max` clauses are specialized
    pub choices: Vec<Choice>,

    /// Currently active groups, as indexes into the root tape
    active_groups: Vec<usize>,
}

/// Represents a tape specialized with a set of choices and active groups
///
/// This is a cheap handle that should be passed by value and cloned
#[derive(Clone)]
pub struct Tape<F>(Arc<InnerTape<F>>);

impl<F: Family> Tape<F> {
    /// Builds a point evaluator from the given `Tape`
    pub fn new_point_evaluator(&self) -> crate::eval::point::PointEval<F> {
        crate::eval::point::PointEval::new(self.clone())
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_interval_evaluator(
        &self,
    ) -> crate::eval::interval::IntervalEval<F> {
        crate::eval::interval::IntervalEval::new(self.clone())
    }

    /// Builds a float evaluator from the given `Tape`
    pub fn new_float_slice_evaluator(
        &self,
    ) -> crate::eval::float_slice::FloatSliceEval<F> {
        crate::eval::float_slice::FloatSliceEval::new(self)
    }

    /// Builds a float evaluator from the given `Tape`
    pub fn new_grad_slice_evaluator(
        &self,
    ) -> crate::eval::grad_slice::GradSliceEval<F> {
        crate::eval::grad_slice::GradSliceEval::new(self)
    }

    /// Simplifies the current tape given a set of choices
    ///
    /// # Panics
    /// The input `choices` must match our internal choice array size
    pub fn simplify(&self, choices: &[Choice]) -> Self {
        assert_eq!(choices.len(), self.tape.choice_count());
        self.clone()
    }
}

impl<E> std::ops::Deref for Tape<E> {
    type Target = InnerTape<E>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> From<TapeData<F>> for Tape<F> {
    fn from(t: TapeData<F>) -> Self {
        let tape = Arc::new(t);
        let choices = vec![Choice::Both; tape.choice_count()];
        let active_groups = (0..tape.data.len()).into_iter().collect();
        let inner = InnerTape {
            tape,
            choices,
            active_groups,
        };
        Self(Arc::new(inner))
    }
}

impl<F> InnerTape<F> {
    /// Iterates over active clauses in the tape in evaluation order
    pub fn iter_asm<'a>(&'a self) -> impl Iterator<Item = Op> + 'a {
        self.active_groups
            .iter()
            .rev()
            .flat_map(|i| self.tape.data[*i].tape.iter().rev())
            .cloned()
    }
    /// Returns the number of active operations in this tape
    ///
    /// Due to inlining and `Load` / `Store` operations, this may be larger than
    /// the expected number of raw arithmetic operations.
    pub fn len(&self) -> usize {
        self.active_groups
            .iter()
            .rev()
            .map(|i| self.tape.data[*i].tape.len())
            .sum()
    }
}
