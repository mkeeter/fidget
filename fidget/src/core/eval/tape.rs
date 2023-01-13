//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    context::{Context, Node},
    eval::{self, Choice, Family},
    ssa::{Op as SsaOp, Tape as SsaTape},
    vm::{Op as VmOp, RegisterAllocator, Tape as VmTape},
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
    pub fn from_ssa(ssa: SsaTape) -> Self {
        let t = Data::from_ssa(ssa, E::REG_LIMIT);
        Self(Arc::new(t), std::marker::PhantomData)
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
/// Under the hood, [`Data`](Self) stores two different representations:
/// - A tape in single static assignment form ([`ssa::Tape`](SsaTape)), which is
///   suitable for use during tape simplification
/// - A tape in register-allocated form ([`vm::Tape`](VmTape)), which can be
///   efficiently evaluated or lowered into machine assembly
#[derive(Default)]
pub struct Data {
    vars: Arc<BTreeMap<String, u32>>,
    asm: VmTape,
}

impl Data {
    /// Returns this tape's mapping of variable names to indexes
    pub fn vars(&self) -> Arc<BTreeMap<String, u32>> {
        self.vars.clone()
    }

    /// Returns the length of the internal `vm::Op` tape
    pub fn len(&self) -> usize {
        self.asm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.asm.is_empty()
    }

    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocate spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.asm.choice_count
    }

    /// Performs register allocation on a [`ssa::Tape`](SsaTape), building a
    /// complete [`Data`](Self).
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let asm = ssa.get_asm(reg_limit);
        Self {
            vars: ssa.vars,
            asm,
        }
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
        // TODO is this necessary?
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
                VmOp::Input(_out, arg) => SsaOp::Input(out, arg as u32),
                VmOp::Var(_out, var) => SsaOp::Var(out, var as u32),
                VmOp::NegReg(_out, arg) => {
                    SsaOp::NegReg(out, workspace.arg(arg))
                }
                VmOp::AbsReg(_out, arg) => {
                    SsaOp::AbsReg(out, workspace.arg(arg))
                }
                VmOp::RecipReg(_out, arg) => {
                    SsaOp::RecipReg(out, workspace.arg(arg))
                }
                VmOp::SqrtReg(_out, arg) => {
                    SsaOp::SqrtReg(out, workspace.arg(arg))
                }
                VmOp::SquareReg(_out, arg) => {
                    SsaOp::SquareReg(out, workspace.arg(arg))
                }
                VmOp::CopyReg(_out, src) => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    match workspace.active(src) {
                        Some(new_src) => SsaOp::CopyReg(out, new_src),
                        None => {
                            workspace.set_active(src, out);
                            continue;
                        }
                    }
                }
                VmOp::AddRegImm(_out, arg, imm) => {
                    SsaOp::AddRegImm(out, workspace.arg(arg), imm)
                }
                VmOp::MulRegImm(_out, arg, imm) => {
                    SsaOp::MulRegImm(out, workspace.arg(arg), imm)
                }
                VmOp::DivRegImm(_out, arg, imm) => {
                    SsaOp::DivRegImm(out, workspace.arg(arg), imm)
                }
                VmOp::DivImmReg(_out, arg, imm) => {
                    SsaOp::DivImmReg(out, workspace.arg(arg), imm)
                }
                VmOp::SubImmReg(_out, arg, imm) => {
                    SsaOp::SubImmReg(out, workspace.arg(arg), imm)
                }
                VmOp::SubRegImm(_out, arg, imm) => {
                    SsaOp::SubRegImm(out, workspace.arg(arg), imm)
                }
                VmOp::MinRegImm(_out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(arg) {
                            Some(new_arg) => SsaOp::CopyReg(out, new_arg),
                            None => {
                                workspace.set_active(arg, out);
                                continue;
                            }
                        },
                        Choice::Right => SsaOp::CopyImm(out, imm),
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MinRegImm(out, workspace.arg(arg), imm)
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                VmOp::MaxRegImm(_out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(arg) {
                            Some(new_arg) => SsaOp::CopyReg(out, new_arg),
                            None => {
                                workspace.set_active(arg, out);
                                continue;
                            }
                        },
                        Choice::Right => SsaOp::CopyImm(out, imm),
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MaxRegImm(out, workspace.arg(arg), imm)
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                VmOp::AddRegReg(_out, lhs, rhs) => SsaOp::AddRegReg(
                    out,
                    workspace.arg(lhs),
                    workspace.arg(rhs),
                ),
                VmOp::MulRegReg(_out, lhs, rhs) => SsaOp::MulRegReg(
                    out,
                    workspace.arg(lhs),
                    workspace.arg(rhs),
                ),
                VmOp::DivRegReg(_out, lhs, rhs) => SsaOp::DivRegReg(
                    out,
                    workspace.arg(lhs),
                    workspace.arg(rhs),
                ),
                VmOp::SubRegReg(_out, lhs, rhs) => SsaOp::SubRegReg(
                    out,
                    workspace.arg(lhs),
                    workspace.arg(rhs),
                ),
                VmOp::MinRegReg(_out, lhs, rhs) => {
                    let c = choice_iter.next().unwrap();
                    match c {
                        Choice::Left => match workspace.active(lhs) {
                            Some(new_lhs) => SsaOp::CopyReg(out, new_lhs),
                            None => {
                                workspace.set_active(lhs, out);
                                continue;
                            }
                        },
                        Choice::Right => match workspace.active(rhs) {
                            Some(new_rhs) => SsaOp::CopyReg(out, new_rhs),
                            None => {
                                workspace.set_active(rhs, out);
                                continue;
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MinRegReg(
                                out,
                                workspace.arg(lhs),
                                workspace.arg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                VmOp::MaxRegReg(_out, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active(lhs) {
                            Some(new_lhs) => SsaOp::CopyReg(out, new_lhs),
                            None => {
                                workspace.set_active(lhs, out);
                                continue;
                            }
                        },
                        Choice::Right => match workspace.active(rhs) {
                            Some(new_rhs) => SsaOp::CopyReg(out, new_rhs),
                            None => {
                                workspace.set_active(rhs, out);
                                continue;
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MaxRegReg(
                                out,
                                workspace.arg(lhs),
                                workspace.arg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                VmOp::CopyImm(_out, imm) => SsaOp::CopyImm(out, imm),
                VmOp::Load(reg, mem) => {
                    let ssa = workspace.bind[reg as usize];
                    workspace.bind[mem as usize] = ssa;
                    workspace.bind[reg as usize] = u32::MAX;
                    continue;
                }
                VmOp::Store(reg, mem) => {
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

    /// Produces an iterator that visits [`vm::Op`](crate::vm::Op) values in
    /// evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = VmOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    /// Pretty-prints the inner SSA tape
    pub fn pretty_print(&self) {
        todo!("oh no")
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
    /// [`vm::Tape`](crate::vm::Tape).
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
