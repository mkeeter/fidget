//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    context::{Context, Node},
    eval::{self, Choice, Family},
    tape::{Op, RegisterAllocator, Tape as VmTape},
    Error,
};
use std::{collections::BTreeMap, sync::Arc};

/// Light-weight handle for tape data, which deferences to
/// [`Data`](Data).
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

/// Safety:
/// The `Tape` contains an `Arc`; the only reason this can't be derived
/// automatically is because it also contains a `PhantomData`.
unsafe impl<R> Send for Tape<R> {}

impl<E: Family> Tape<E> {
    pub fn new(data: Data) -> Self {
        Self(Arc::new(data), std::marker::PhantomData)
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

impl<E: crate::eval::Family> TryFrom<(Node, Context)> for Tape<E> {
    type Error = Error;
    fn try_from((node, context): (Node, Context)) -> Result<Self, Self::Error> {
        context.get_tape(node)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`Data`](Self) stores a tape in register-allocated form
/// ([`tape::Tape`](VmTape)), which can be efficiently evaluated or lowered into
/// machine assembly
#[derive(Default)]
pub struct Data {
    vars: Arc<BTreeMap<String, u32>>,
    asm: VmTape,
}

impl Data {
    pub fn new(asm: VmTape, vars: BTreeMap<String, u32>) -> Self {
        Self {
            vars: Arc::new(vars),
            asm,
        }
    }

    /// Returns this tape's mapping of variable names to indexes
    pub fn vars(&self) -> Arc<BTreeMap<String, u32>> {
        self.vars.clone()
    }

    /// Returns the length of the internal `tape::Op` tape
    pub fn len(&self) -> usize {
        self.asm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.asm.is_empty()
    }

    /// Pushes a single operation to an empty type.
    ///
    /// This is only used when the tape is a single constant.
    ///
    /// # Panics
    /// If the tape is not empty.
    pub(crate) fn push(&mut self, v: Op) {
        assert!(self.asm.is_empty());
        self.asm.push(v);
    }

    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocate spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.asm.choice_count
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Returns the register limit of the VM tape
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
        tape: Data,
    ) -> Result<Self, Error> {
        if choices.len() != self.choice_count() {
            return Err(Error::BadChoiceSlice(
                choices.len(),
                self.choice_count(),
            ));
        }
        let reg_limit = self.asm.reg_limit();

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        workspace.reset_with_storage(
            reg_limit,
            self.asm.len(),
            self.slot_count(),
            tape.asm,
        );

        // The tape is constructed so that the output slot is first
        assert_eq!(self.asm.tape[0].output().unwrap(), 0);
        workspace.set_active(self.asm.tape[0].output().unwrap(), 0);
        workspace.count += 1;

        // Perform in-line translation from pseudo-linear to SSA form
        let mut choice_count = 0;
        let mut choice_iter = choices.iter().rev();
        for op in self.asm.iter() {
            // Skip if this operation isn't active (which we check by seeing
            // whether its output is bound)
            let out = if let Some(out) = op.output() {
                if workspace.active(out).is_none() {
                    for _ in 0..op.choice_count() {
                        choice_iter.next().unwrap();
                    }
                    continue;
                }
                workspace.out(out)
            } else {
                // Not used for Load/Store operations
                u32::MAX
            };

            let op = match *op {
                Op::Input(_out, arg) => Op::Input(out, arg),
                Op::Var(_out, var) => Op::Var(out, var as u32),
                Op::NegReg(_out, arg) => Op::NegReg(out, workspace.arg(arg)),
                Op::AbsReg(_out, arg) => Op::AbsReg(out, workspace.arg(arg)),
                Op::RecipReg(_out, arg) => {
                    Op::RecipReg(out, workspace.arg(arg))
                }
                Op::SqrtReg(_out, arg) => Op::SqrtReg(out, workspace.arg(arg)),
                Op::SquareReg(_out, arg) => {
                    Op::SquareReg(out, workspace.arg(arg))
                }
                Op::CopyReg(_out, src) => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    match workspace.active(src) {
                        Some(new_src) => Op::CopyReg(out, new_src),
                        None => {
                            workspace.set_active(src, out);
                            continue;
                        }
                    }
                }
                Op::AddRegImm(_out, arg, imm) => {
                    Op::AddRegImm(out, workspace.arg(arg), imm)
                }
                Op::MulRegImm(_out, arg, imm) => {
                    Op::MulRegImm(out, workspace.arg(arg), imm)
                }
                Op::DivRegImm(_out, arg, imm) => {
                    Op::DivRegImm(out, workspace.arg(arg), imm)
                }
                Op::DivImmReg(_out, arg, imm) => {
                    Op::DivImmReg(out, workspace.arg(arg), imm)
                }
                Op::SubImmReg(_out, arg, imm) => {
                    Op::SubImmReg(out, workspace.arg(arg), imm)
                }
                Op::SubRegImm(_out, arg, imm) => {
                    Op::SubRegImm(out, workspace.arg(arg), imm)
                }
                Op::MinRegImm(_out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(arg) {
                            Some(new_arg) => Op::CopyReg(out, new_arg),
                            None => {
                                workspace.set_active(arg, out);
                                continue;
                            }
                        },
                        Choice::Right => Op::CopyImm(out, imm),
                        Choice::Both => {
                            choice_count += 1;
                            Op::MinRegImm(out, workspace.arg(arg), imm)
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                Op::MaxRegImm(_out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(arg) {
                            Some(new_arg) => Op::CopyReg(out, new_arg),
                            None => {
                                workspace.set_active(arg, out);
                                continue;
                            }
                        },
                        Choice::Right => Op::CopyImm(out, imm),
                        Choice::Both => {
                            choice_count += 1;
                            Op::MaxRegImm(out, workspace.arg(arg), imm)
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                Op::AddRegReg(_out, lhs, rhs) => {
                    Op::AddRegReg(out, workspace.arg(lhs), workspace.arg(rhs))
                }
                Op::MulRegReg(_out, lhs, rhs) => {
                    Op::MulRegReg(out, workspace.arg(lhs), workspace.arg(rhs))
                }
                Op::DivRegReg(_out, lhs, rhs) => {
                    Op::DivRegReg(out, workspace.arg(lhs), workspace.arg(rhs))
                }
                Op::SubRegReg(_out, lhs, rhs) => {
                    Op::SubRegReg(out, workspace.arg(lhs), workspace.arg(rhs))
                }
                Op::MinRegReg(_out, lhs, rhs) => {
                    let c = choice_iter.next().unwrap();
                    match c {
                        Choice::Left => match workspace.active(lhs) {
                            Some(new_lhs) => Op::CopyReg(out, new_lhs),
                            None => {
                                workspace.set_active(lhs, out);
                                continue;
                            }
                        },
                        Choice::Right => match workspace.active(rhs) {
                            Some(new_rhs) => Op::CopyReg(out, new_rhs),
                            None => {
                                workspace.set_active(rhs, out);
                                continue;
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            Op::MinRegReg(
                                out,
                                workspace.arg(lhs),
                                workspace.arg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                Op::MaxRegReg(_out, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(lhs) {
                            Some(new_lhs) => Op::CopyReg(out, new_lhs),
                            None => {
                                workspace.set_active(lhs, out);
                                continue;
                            }
                        },
                        Choice::Right => match workspace.active(rhs) {
                            Some(new_rhs) => Op::CopyReg(out, new_rhs),
                            None => {
                                workspace.set_active(rhs, out);
                                continue;
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            Op::MaxRegReg(
                                out,
                                workspace.arg(lhs),
                                workspace.arg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                Op::CopyImm(_out, imm) => Op::CopyImm(out, imm),
                Op::Load(reg, mem) => {
                    let ssa = workspace.bind[reg as usize];
                    workspace.bind[mem as usize] = ssa;
                    workspace.bind[reg as usize] = u32::MAX;
                    continue;
                }
                Op::Store(reg, mem) => {
                    let ssa = workspace.bind[mem as usize];
                    workspace.bind[reg as usize] = ssa;
                    workspace.bind[mem as usize] = u32::MAX;
                    continue;
                }
            };
            workspace.alloc.op(op);
        }

        let mut asm_tape = workspace.alloc.finalize();
        asm_tape.choice_count = choice_count;

        Ok(Data {
            vars: self.vars.clone(),
            asm: asm_tape,
        })
    }

    /// Produces an iterator that visits [`tape::Op`](crate::tape::Op) values in
    /// evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = Op> + '_ {
        self.asm.iter().cloned().rev()
    }

    /// Pretty-prints the inner pseudo-assembly tape
    pub fn pretty_print(&self) {
        self.asm.pretty_print();
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Data structures used during [`Tape::simplify`]
///
/// This is exposed to minimize reallocations in hot loops.
pub struct Workspace {
    pub alloc: RegisterAllocator,
    pub bind: Vec<u32>,
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
    fn active(&self, i: u8) -> Option<u32> {
        Some(self.bind[i as usize]).filter(|o| *o != u32::MAX)
    }

    fn out(&mut self, i: u8) -> u32 {
        let out = self.active(i).unwrap();
        self.bind[i as usize] = u32::MAX;
        out
    }

    fn arg(&mut self, i: u8) -> u32 {
        if self.bind[i as usize] == u32::MAX {
            self.bind[i as usize] = self.count;
            self.count += 1;
        }
        self.bind[i as usize]
    }

    fn set_active(&mut self, i: u8, bind: u32) {
        self.bind[i as usize] = bind;
    }

    /// Resets the workspace, preserving allocations
    pub fn reset(
        &mut self,
        num_registers: u8,
        tape_len: usize,
        slot_count: usize,
    ) {
        self.reset_with_storage(
            num_registers,
            tape_len,
            slot_count,
            Default::default(),
        );
    }

    /// Resets the workspace, preserving allocations and claiming the given
    /// [`tape::Tape`](crate::tape::Tape).
    pub fn reset_with_storage(
        &mut self,
        num_registers: u8,
        tape_len: usize,
        slot_count: usize,
        tape: VmTape,
    ) {
        self.alloc.reset_with_storage(num_registers, tape_len, tape);
        self.bind.fill(u32::MAX);
        self.bind.resize(slot_count, u32::MAX);
        self.count = 0;
    }
}
