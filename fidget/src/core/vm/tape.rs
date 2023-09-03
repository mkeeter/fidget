use std::{collections::BTreeMap, sync::Arc};

use crate::{
    eval::{
        float_slice::FloatSliceEvalStorage, grad_slice::GradSliceEvalStorage,
        interval::IntervalEvalStorage, point::PointEvalStorage, Family,
        FloatSliceEval, GradSliceEval, IntervalEval, PointEval,
    },
    vm::{ChoiceMask, Op},
};

/// Evaluation tape, which is an ordered set of functions
///
/// It is parameterized by an [`Family`](Family) type, which sets the register
/// count of the inner VM tapes.
///
/// This is a heavy-weight value and should not change once constructed;
/// consider wrapping it in an `Arc` for shared light-weight use.
pub struct TapeData<F: Family> {
    pub data: F::TapeData,

    /// Group metadata
    pub groups: Vec<GroupMetadata<F>>,

    /// Size of the choice array
    choice_array_size: usize,

    /// Number of slots used by the tape
    slot_count: usize,

    /// Mapping from variable names (in the original
    /// [`Context`](crate::context::Context)) to indexes in the variable array
    /// used during evaluation.
    vars: BTreeMap<String, u32>,
}

impl<F: Family> TapeData<F> {
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
    /// Returns the size of the choice array (as a number of `u64` words)
    pub fn choice_array_size(&self) -> usize {
        self.choice_array_size
    }

    /// Returns the total number of operations in the tape
    ///
    /// This is the sum of all operations in the node groups.  Due to inlining
    /// and the insertion of `Load` and `Store` operations, it may be larger
    /// than the raw number of arithmetic operations in the expression!
    pub fn len(&self) -> usize {
        self.groups.iter().map(|c| c.len).sum()
    }
}

impl<F: Family> TapeData<F> {
    /// Returns the register limit used when planning this tape
    pub fn reg_limit(&self) -> u8 {
        F::REG_LIMIT
    }
}

// TODO: move this elsewhere, since it's only used during construction?
/// A tape alongside the choices which lead it to being selected
///
/// Specifically, the tape is active if _any of_ `self.choices` is true
///
/// If the tape is inactive, then we should clear choices given by `self.clear`
#[derive(Debug)]
pub struct ChoiceTape {
    /// List of instructions in reverse-evaluation order
    pub tape: Vec<Op>,

    /// If any of the given bits are set, then this group is active
    ///
    /// As a special case, the always-selected (root) tape is represented with
    /// an empty vector.
    pub choices: Vec<ChoiceMask>,

    /// When this group is inactive, `AND` the given choice indices with masks
    pub clear: Vec<(usize, u64)>,
}

#[derive(Debug)]
pub struct GroupMetadata<F: Family> {
    /// Per-family group metadata
    ///
    /// For example, in the VM evaluator, this is a range into the data `Vec`
    /// that covers the group's operations.
    pub data: F::GroupMetadata,

    /// Number of operations in this group
    pub len: usize,

    /// Array of choice indices which are set when a group is active
    pub choice_mask_range: Vec<ChoiceMask>,

    /// When a group is inactive, apply the associated masks (with `AND`)
    pub clear_range: Vec<(usize, u64)>,
}

impl<F: Family> TapeData<F> {
    /// Builds a new tape, built from many operation groups
    ///
    /// `data` should be ordered in reverse-evaluation order (root to leaf)
    pub fn new(data: Vec<ChoiceTape>, vars: BTreeMap<String, u32>) -> Self {
        let choice_array_size = data
            .iter()
            .flat_map(|t| t.choices.iter())
            .map(|c| c.index)
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

        let (tape_data, group_data) =
            F::build(slot_count, choice_array_size, &data);
        let mut groups = vec![];
        for (c, g) in data.iter().zip(group_data) {
            groups.push(GroupMetadata {
                choice_mask_range: c.choices.clone(),
                clear_range: c.clear.clone(),
                len: c.tape.len(),
                data: g,
            });
        }

        Self {
            groups,
            data: tape_data,
            vars,
            choice_array_size,
            slot_count,
        }
    }
}

/// Represents choices and active groups used to specialize a tape
///
/// This is a heavy-weight data structure that's typically wrapped in a
/// [`Tape`] for ease of use.
#[derive(Default)]
pub struct TapeSpecialization {
    /// Currently active groups, as indexes into the [`TapeData::groups`] array
    active_groups: Vec<usize>, // TODO should this be ranges instead?
}

/// Represents a tape specialized with a set of choices and active groups
///
/// This is a cheap handle that can be passed by value and cloned
#[derive(Clone)]
pub struct Tape<F: Family>(Arc<(Arc<TapeData<F>>, TapeSpecialization)>);

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

impl<F: Family> Tape<F> {
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
        // TODO: move this to the family-specific code?
        self.data().slot_count()
    }

    /// Simplifies the current tape given a set of choices
    ///
    /// # Panics
    /// The input `choices` must match our internal choice array size
    pub fn simplify(&self, choices: &mut [u64]) -> Self {
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
        choices: &mut [u64],
        mut prev: TapeSpecialization,
    ) -> Self {
        prev.active_groups.clear();
        let data = self.data().as_ref();
        for &g in &(self.0).1.active_groups {
            let metadata = &data.groups[g];
            let choice_sel = &metadata.choice_mask_range;
            let still_active = choice_sel.is_empty()
                || choice_sel
                    .into_iter()
                    .any(|c| choices[c.index as usize] & c.mask != 0);
            if still_active {
                prev.active_groups.push(g);
            } else {
                for (i, mask) in metadata.clear_range.iter() {
                    choices[*i] &= mask;
                }
            }
        }
        Self(Arc::new((self.data().clone(), prev)))
    }

    /// Moves internal allocations into a new object, leaving this one empty
    pub fn take(self) -> Option<TapeSpecialization> {
        Arc::try_unwrap(self.0).ok().map(|t| t.1)
    }

    /// Returns the number of active operations in this tape
    ///
    /// Due to inlining and `Load` / `Store` operations, this may be larger than
    /// the expected number of raw arithmetic operations.
    pub fn len(&self) -> usize {
        self.active_groups()
            .iter()
            .map(|i| self.data().groups[*i].len)
            .sum()
    }
}

impl<F: Family> From<TapeData<F>> for Tape<F> {
    fn from(tape: TapeData<F>) -> Self {
        // Initially, all groups are active!
        let active_groups: Vec<usize> =
            (0..tape.groups.len()).into_iter().collect();
        Self(Arc::new((
            Arc::new(tape),
            TapeSpecialization { active_groups },
        )))
    }
}
