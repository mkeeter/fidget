//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    compiler::{RegOp, RegTape, RegisterAllocator, SsaOp, SsaTape},
    context::{Context, Node},
    eval::{self, Choice, Family},
    Error,
};
use std::{collections::HashMap, sync::Arc};

/// Light-weight handle for tape data, which deferences to
/// [`Data`].
///
/// This can be passed by value and cloned.
///
/// It is parameterized by an [`Family`](Family) type, which sets the register
/// count of the inner VM tape.
pub struct Tape<R>(Arc<Data>, std::marker::PhantomData<*const R>);

impl<R> Clone for Tape<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), std::marker::PhantomData)
    }
}

// SAFETY: the tape contains a single `Arc`, which is already `Send + Sync`;
// the only reason this can't be derived automatically is due to the
// `PhantomData` and Rust limitations.
unsafe impl<R> Send for Tape<R> {}
unsafe impl<R> Sync for Tape<R> {}

impl<E: Family> Tape<E> {
    /// Build a new tape for the given node
    pub fn new(context: &Context, node: Node) -> Result<Self, Error> {
        let ssa = SsaTape::new(context, node)?;
        let asm = RegTape::new(&ssa, E::REG_LIMIT);
        Ok(Self(Arc::new(Data { ssa, asm }), std::marker::PhantomData))
    }

    /// Simplifies a tape based on the array of choices
    ///
    /// The choice slice must be the same size as
    /// [`self.choice_count()`](Data::choice_count),
    /// which should be ensured by the caller.
    pub fn simplify(&self, choices: &[Choice]) -> Result<Self, Error> {
        self.simplify_with(choices, &mut Default::default(), Default::default())
    }

    /// Simplifies a tape, reusing workspace and allocations
    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        prev: Data,
    ) -> Result<Self, Error> {
        self.0
            .simplify_with(choices, workspace, prev)
            .map(Arc::new)
            .map(|t| Tape(t, std::marker::PhantomData))
    }

    /// Tries to claim the inner [`Data`]
    ///
    /// This will fail if there are multiple `Tape` objects sharing the `Data`.
    pub fn take(self) -> Option<Data> {
        Arc::try_unwrap(self.0).ok()
    }

    /// Builds a point evaluator from the given `Tape`
    pub fn new_point_evaluator(&self) -> eval::point::PointEval<E> {
        eval::point::PointEval::new(self)
    }

    /// Builds an interval evaluator from the given `Tape`
    pub fn new_interval_evaluator(&self) -> eval::interval::IntervalEval<E> {
        eval::interval::IntervalEval::new(self)
    }

    /// Builds an interval evaluator from the given `Tape`, reusing storage
    pub fn new_interval_evaluator_with_storage(
        &self,
        storage: eval::interval::IntervalEvalStorage<E>,
    ) -> eval::interval::IntervalEval<E> {
        eval::interval::IntervalEval::new_with_storage(self, storage)
    }

    /// Builds a float evaluator from the given `Tape`
    pub fn new_float_slice_evaluator(
        &self,
    ) -> eval::float_slice::FloatSliceEval<E> {
        eval::float_slice::FloatSliceEval::new(self)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    pub fn new_float_slice_evaluator_with_storage(
        &self,
        storage: eval::float_slice::FloatSliceEvalStorage<E>,
    ) -> eval::float_slice::FloatSliceEval<E> {
        eval::float_slice::FloatSliceEval::new_with_storage(self, storage)
    }

    /// Builds a grad slice evaluator from the given `Tape`
    pub fn new_grad_slice_evaluator(
        &self,
    ) -> eval::grad_slice::GradSliceEval<E> {
        eval::grad_slice::GradSliceEval::new(self)
    }

    /// Builds a float slice evaluator from the given `Tape`, reusing storage
    pub fn new_grad_slice_evaluator_with_storage(
        &self,
        storage: eval::grad_slice::GradSliceEvalStorage<E>,
    ) -> eval::grad_slice::GradSliceEval<E> {
        eval::grad_slice::GradSliceEval::new_with_storage(self, storage)
    }
}

impl<E> std::ops::Deref for Tape<E> {
    type Target = Data;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`Data`](Self) stores two different representations:
/// - A tape in single static assignment form ([`ssa::Tape`](SsaTape)), which is
///   suitable for use during tape simplification
/// - A tape in register-allocated form ([`vm::Tape`](RegTape)), which can be
///   efficiently evaluated or lowered into machine assembly
#[derive(Default)]
pub struct Data {
    ssa: SsaTape,
    asm: RegTape,
}

impl Data {
    /// Returns this tape's mapping of variable names to indexes
    pub fn vars(&self) -> Arc<HashMap<String, u32>> {
        self.ssa.vars.clone()
    }

    /// Returns the length of the internal VM tape
    pub fn len(&self) -> usize {
        self.asm.len()
    }

    /// Returns true if the internal VM tape is empty
    pub fn is_empty(&self) -> bool {
        self.asm.is_empty()
    }

    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocate spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.ssa.choice_count
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    /// Returns the number of variables used in this tape
    pub fn var_count(&self) -> usize {
        self.ssa.vars.len()
    }

    /// Returns the register limit of the VM tape
    ///
    /// Note that this may exceed the slot count on particularly short (or
    /// easy-to-plan) tapes.
    pub fn reg_limit(&self) -> u8 {
        self.asm.reg_limit()
    }

    /// Simplifies both inner tapes, using the provided choice array
    ///
    /// To minimize allocations, this function takes a [`Workspace`](Workspace)
    /// _and_ spare [`Data`](Data); it will reuse those allocations.
    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        mut tape: Data,
    ) -> Result<Self, Error> {
        if choices.len() != self.choice_count() {
            return Err(Error::BadChoiceSlice(
                choices.len(),
                self.choice_count(),
            ));
        }
        let reg_limit = self.asm.reg_limit();
        tape.ssa.reset();

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        workspace.reset_with_storage(reg_limit, self.ssa.tape.len(), tape.asm);

        let mut choice_count = 0;

        // The tape is constructed so that the output slot is first
        assert_eq!(self.ssa.tape[0].output(), 0);
        workspace.set_active(self.ssa.tape[0].output(), 0);
        workspace.count += 1;

        // Other iterators to consume various arrays in order
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = tape.ssa.tape;

        for mut op in self.ssa.tape.iter().cloned() {
            let index = op.output();

            if workspace.active(index).is_none() {
                for _ in 0..op.choice_count() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Because we reassign nodes when they're used as an *input*
            // (while walking the tape in reverse), this node must have been
            // assigned already.
            let new_index = workspace.active(index).unwrap();

            match &mut op {
                SsaOp::Input(index, ..)
                | SsaOp::Var(index, ..)
                | SsaOp::CopyImm(index, ..) => {
                    *index = new_index;
                }
                SsaOp::NegReg(index, arg)
                | SsaOp::AbsReg(index, arg)
                | SsaOp::RecipReg(index, arg)
                | SsaOp::SqrtReg(index, arg)
                | SsaOp::SquareReg(index, arg) => {
                    *index = new_index;
                    *arg = workspace.get_or_insert_active(*arg);
                }
                SsaOp::CopyReg(index, src) => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    match workspace.active(*src) {
                        Some(new_src) => {
                            *index = new_index;
                            *src = new_src;
                        }
                        None => {
                            workspace.set_active(*src, new_index);
                            continue;
                        }
                    }
                }
                SsaOp::MinRegImm(index, arg, imm)
                | SsaOp::MaxRegImm(index, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(*arg) {
                            Some(new_arg) => {
                                op = SsaOp::CopyReg(new_index, new_arg);
                            }
                            None => {
                                workspace.set_active(*arg, new_index);
                                continue;
                            }
                        },
                        Choice::Right => {
                            op = SsaOp::CopyImm(new_index, *imm);
                        }
                        Choice::Both => {
                            choice_count += 1;
                            *index = new_index;
                            *arg = workspace.get_or_insert_active(*arg);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                SsaOp::MinRegReg(index, lhs, rhs)
                | SsaOp::MaxRegReg(index, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(*lhs) {
                            Some(new_lhs) => {
                                op = SsaOp::CopyReg(new_index, new_lhs);
                            }
                            None => {
                                workspace.set_active(*lhs, new_index);
                                continue;
                            }
                        },
                        Choice::Right => match workspace.active(*rhs) {
                            Some(new_rhs) => {
                                op = SsaOp::CopyReg(new_index, new_rhs);
                            }
                            None => {
                                workspace.set_active(*rhs, new_index);
                                continue;
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            *index = new_index;
                            *lhs = workspace.get_or_insert_active(*lhs);
                            *rhs = workspace.get_or_insert_active(*rhs);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                SsaOp::AddRegReg(index, lhs, rhs)
                | SsaOp::MulRegReg(index, lhs, rhs)
                | SsaOp::SubRegReg(index, lhs, rhs)
                | SsaOp::DivRegReg(index, lhs, rhs) => {
                    *index = new_index;
                    *lhs = workspace.get_or_insert_active(*lhs);
                    *rhs = workspace.get_or_insert_active(*rhs);
                }
                SsaOp::AddRegImm(index, arg, _imm)
                | SsaOp::MulRegImm(index, arg, _imm)
                | SsaOp::SubRegImm(index, arg, _imm)
                | SsaOp::SubImmReg(index, arg, _imm)
                | SsaOp::DivRegImm(index, arg, _imm)
                | SsaOp::DivImmReg(index, arg, _imm) => {
                    *index = new_index;
                    *arg = workspace.get_or_insert_active(*arg);
                }
            }
            workspace.alloc.op(op);
            ops_out.push(op);
        }

        assert_eq!(workspace.count as usize, ops_out.len());
        let asm_tape = workspace.alloc.finalize();

        Ok(Data {
            ssa: SsaTape {
                tape: ops_out,
                choice_count,
                vars: self.ssa.vars.clone(),
            },
            asm: asm_tape,
        })
    }

    /// Produces an iterator that visits [`RegOp`](crate::compiler::RegOp)
    /// values in evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = RegOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    /// Pretty-prints the inner SSA tape
    pub fn pretty_print(&self) {
        self.ssa.pretty_print();
        for a in self.iter_asm() {
            println!("{a:?}");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Data structures used during [`Tape::simplify`]
///
/// This is exposed to minimize reallocations in hot loops.
pub struct Workspace {
    /// Register allocator
    pub alloc: RegisterAllocator,

    /// Current bindings from SSA variables to registers
    pub bind: Vec<u32>,

    /// Number of active SSA bindings
    ///
    /// This value is monotonically increasing; each SSA variable gets the next
    /// value if it is unassigned when encountered.
    count: u32,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            alloc: RegisterAllocator::empty(),
            bind: vec![],
            count: 0,
        }
    }
}

impl Workspace {
    fn active(&self, i: u32) -> Option<u32> {
        if self.bind[i as usize] != u32::MAX {
            Some(self.bind[i as usize])
        } else {
            None
        }
    }

    fn get_or_insert_active(&mut self, i: u32) -> u32 {
        if self.bind[i as usize] == u32::MAX {
            self.bind[i as usize] = self.count;
            self.count += 1;
        }
        self.bind[i as usize]
    }

    fn set_active(&mut self, i: u32, bind: u32) {
        self.bind[i as usize] = bind;
    }

    /// Resets the workspace, preserving allocations
    pub fn reset(&mut self, num_registers: u8, tape_len: usize) {
        self.alloc.reset(num_registers, tape_len);
        self.bind.fill(u32::MAX);
        self.bind.resize(tape_len, u32::MAX);
        self.count = 0;
    }

    /// Resets the workspace, preserving allocations and claiming the given
    /// [`RegTape`].
    pub fn reset_with_storage(
        &mut self,
        num_registers: u8,
        tape_len: usize,
        tape: RegTape,
    ) {
        self.alloc.reset_with_storage(num_registers, tape_len, tape);
        self.bind.fill(u32::MAX);
        self.bind.resize(tape_len, u32::MAX);
        self.count = 0;
    }
}
