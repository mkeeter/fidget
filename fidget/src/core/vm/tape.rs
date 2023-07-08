use std::{collections::BTreeMap, sync::Arc};

use crate::{
    eval::{
        float_slice::FloatSliceEvalStorage, grad_slice::GradSliceEvalStorage,
        interval::IntervalEvalStorage, point::PointEvalStorage, Family,
        FloatSliceEval, GradSliceEval, IntervalEval, PointEval,
    },
    vm::{ChoiceIndex, Op},
};

/// Evaluation tape, which is an ordered set of functions
///
/// It is parameterized by an [`Family`](Family) type, which sets the register
/// count of the inner VM tapes.
///
/// This is a heavy-weight value and should not change once constructed;
/// consider wrapping it in an `Arc` for shared light-weight use.
#[derive(Debug)]
pub struct TapeData<F> {
    /// The tape groups are stored in reverse order, such that the root of the
    /// tree is the first item in the first tape
    pub data: Vec<ChoiceTape>,

    /// Size of the choice array
    choice_array_size: usize,

    /// Number of slots used by the tape
    slot_count: usize,

    /// Mapping from variable names (in the original
    /// [`Context`](crate::context::Context)) to indexes in the variable array
    /// used during evaluation.
    vars: BTreeMap<String, u32>,

    _p: std::marker::PhantomData<fn() -> F>,
}

impl<F> std::ops::Index<usize> for TapeData<F> {
    type Output = ChoiceTape;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
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
    pub fn choice_array_size(&self) -> usize {
        self.choice_array_size
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
///
/// Specifically, the tape is active if _any of_ `self.choices` is true **and**
/// `self.skip` is false (or not present)
#[derive(Debug)]
pub struct ChoiceTape {
    /// List of instructions in reverse-evaluation order
    pub tape: Vec<Op>,

    /// Array of choice indices which are set when this tape is active
    ///
    /// As a special case, the always-selected (root) tape is represented with
    /// an empty vector.
    pub choices: Vec<ChoiceIndex>,

    /// When this group is inactive, clear the given choice ranges
    pub clear: Vec<std::ops::Range<usize>>,
}

impl<F: Family> TapeData<F> {
    /// Builds a new tape, built from many operation groups
    pub fn new(data: Vec<ChoiceTape>, vars: BTreeMap<String, u32>) -> Self {
        let choice_array_size = data
            .iter()
            .flat_map(|t| t.choices.iter())
            .map(|c| c.index + c.bit / 8)
            .max()
            .map(|i| i + 1)
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
            choice_array_size,
            slot_count,
            _p: std::marker::PhantomData,
        }
    }
}

/// Represents choices and active groups used to specialize a tape
///
/// This is a heavy-weight data structure that's typically wrapped in a
/// [`Tape`] for ease of use.
#[derive(Default)]
pub struct TapeSpecialization {
    /// Currently active groups, as indexes into the root tape
    active_groups: Vec<usize>,
}

/// Represents a tape specialized with a set of choices and active groups
///
/// This is a cheap handle that can be passed by value and cloned
#[derive(Clone)]
pub struct Tape<F>(Arc<(Arc<TapeData<F>>, TapeSpecialization)>);

impl<F: Family> Tape<F> {
    /// Builds a point evaluator from the given `Tape`
    pub fn new_point_evaluator(&self) -> PointEval<F> {
        PointEval::new(self.clone())
    }

    /// Builds a point evaluator from the given `Tape`
    pub fn new_point_evaluator_with_storage(
        &self,
        storage: PointEvalStorage<F>,
    ) -> PointEval<F> {
        PointEval::new_with_storage(self.clone(), storage)
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_interval_evaluator(&self) -> IntervalEval<F> {
        IntervalEval::new(self.clone())
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_interval_evaluator_with_storage(
        &self,
        storage: IntervalEvalStorage<F>,
    ) -> IntervalEval<F> {
        IntervalEval::new_with_storage(self.clone(), storage)
    }

    /// Builds a float evaluator from the given `Tape`
    pub fn new_float_slice_evaluator(&self) -> FloatSliceEval<F> {
        FloatSliceEval::new(self.clone())
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_float_slice_evaluator_with_storage(
        &self,
        storage: FloatSliceEvalStorage<F>,
    ) -> FloatSliceEval<F> {
        FloatSliceEval::new_with_storage(self.clone(), storage)
    }

    /// Builds a float evaluator from the given `Tape`
    pub fn new_grad_slice_evaluator(&self) -> GradSliceEval<F> {
        GradSliceEval::new(self.clone())
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_grad_slice_evaluator_with_storage(
        &self,
        storage: GradSliceEvalStorage<F>,
    ) -> GradSliceEval<F> {
        GradSliceEval::new_with_storage(self.clone(), storage)
    }
}

impl<F> Tape<F> {
    /// Returns the size of the choice array associated with the tape
    ///
    /// Note that not all choices may be active (depending on tape
    /// specialization), but the caller should operate using the full count.
    pub fn choice_array_size(&self) -> usize {
        self.data().choice_array_size()
    }

    /// Returns the number of variables associated with the root tape
    ///
    /// Note that not all variables may be active (depending on tape
    /// specialization), but the caller should operate using the full count.
    pub fn var_count(&self) -> usize {
        self.data().var_count()
    }

    /// Returns the number of slots associated with the root tape
    ///
    /// Note that not all slot may be active (depending on tape specialization),
    /// but the caller should operate using the full count.
    pub fn slot_count(&self) -> usize {
        self.data().slot_count()
    }

    /// Simplifies the current tape given a set of choices
    ///
    /// # Panics
    /// The input `choices` must match our internal choice array size
    pub fn simplify(&self, choices: &mut [u8]) -> Self {
        self.simplify_with(choices, TapeSpecialization::default())
    }

    /// Returns the inner tape data
    pub fn data(&self) -> &Arc<TapeData<F>> {
        &(self.0).0
    }

    /// Returns the inner tape data
    pub fn active_groups(&self) -> &[usize] {
        &(self.0).1.active_groups
    }

    /// Simplifies a tape, reusing allocations from an [`TapeSpecialization`]
    pub fn simplify_with(
        &self,
        choices: &mut [u8],
        mut prev: TapeSpecialization,
    ) -> Self {
        prev.active_groups.clear();
        for &g in &(self.0).1.active_groups {
            let choice_tape = &self.data()[g];
            let choice_sel = &choice_tape.choices;
            let still_active = choice_sel.is_empty()
                || choice_sel
                    .iter()
                    .any(|&c| choices[c.end()] & (1 << (c.bit % 8)) != 0);
            if still_active {
                prev.active_groups.push(g);
            } else {
                for clear_range in choice_tape.clear.iter() {
                    for c in &mut choices[clear_range.clone()] {
                        *c = 0;
                    }
                }
            }
        }
        Self(Arc::new((self.data().clone(), prev)))
    }

    /// Moves internal allocations into a new object, leaving this one empty
    pub fn take(self) -> Option<TapeSpecialization> {
        Arc::try_unwrap(self.0).ok().map(|t| t.1)
    }

    /// Iterates over active clauses in the tape in evaluation order
    pub fn iter_asm<'a>(&'a self) -> impl Iterator<Item = Op> + 'a {
        self.active_groups()
            .iter()
            .rev()
            .flat_map(|i| self.data()[*i].tape.iter().rev())
            .cloned()
    }
    /// Returns the number of active operations in this tape
    ///
    /// Due to inlining and `Load` / `Store` operations, this may be larger than
    /// the expected number of raw arithmetic operations.
    pub fn len(&self) -> usize {
        self.active_groups()
            .iter()
            .map(|i| self.data()[*i].tape.len())
            .sum()
    }
}

impl<F> From<TapeData<F>> for Tape<F> {
    fn from(tape: TapeData<F>) -> Self {
        let active_groups: Vec<usize> =
            (0..tape.data.len()).into_iter().collect();
        Self(Arc::new((
            Arc::new(tape),
            TapeSpecialization { active_groups },
        )))
    }
}
