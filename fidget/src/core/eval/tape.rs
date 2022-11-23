//! Dual-use tapes for use during evaluation or further compilation
use crate::{
    eval::{Choice, Eval},
    ssa::{Op as SsaOp, Tape as SsaTape},
    vm::{Op as VmOp, RegisterAllocator, Tape as VmTape},
    Error,
};
use std::{collections::BTreeMap, sync::Arc};

/// Light-weight handle for tape data, which deferences to
/// [`TapeData`](TapeData).
///
/// This can be passed by value and cloned.
///
/// It is parameterized by an [`Eval`](Eval) type, which sets the register
/// count of the inner VM tape.
#[derive(Clone)]
pub struct Tape<R>(Arc<TapeData>, std::marker::PhantomData<*const R>);

/// Safety:
/// The `Tape` contains an `Arc`; the only reason this can't be derived
/// automatically is because it also contains a `PhantomData`.
unsafe impl<R> Send for Tape<R> {}

impl<E: Eval> Tape<E> {
    pub fn from_ssa(ssa: SsaTape) -> Self {
        let t = TapeData::from_ssa(ssa, E::REG_LIMIT);
        Self(Arc::new(t), std::marker::PhantomData)
    }

    /// Simplifies a tape based on the array of choices
    ///
    /// The choice slice must be the same size as
    /// [`self.choice_count()`](TapeData::choice_count),
    /// which should be ensured by the caller.
    pub fn simplify(&self, choices: &[Choice]) -> Result<Self, Error> {
        self.simplify_with(choices, &mut Default::default(), Default::default())
    }

    /// Simplifies a tape, reusing workspace and allocations
    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        prev: TapeData,
    ) -> Result<Self, Error> {
        self.0
            .simplify_with(choices, workspace, prev)
            .map(Arc::new)
            .map(|t| Tape(t, std::marker::PhantomData))
    }

    pub fn take(self) -> Option<TapeData> {
        Arc::try_unwrap(self.0).ok()
    }
}

impl<E> std::ops::Deref for Tape<E> {
    type Target = TapeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`TapeData`](Self) stores two different representations:
/// - A tape in single static assignment form ([`ssa::Tape`}(crate::ssa::Tape)),
///   which is suitable for use during tape simplification
/// - A tape in register-allocated form ([`vm::Tape`](crate::vm::Tape)), which
///   can be efficiently evaluated or lowered into machine assembly
#[derive(Default)]
pub struct TapeData {
    ssa: SsaTape,
    asm: VmTape,
}

impl TapeData {
    /// Empties out the inner tapes, retaining their allocations
    pub fn reset(&mut self) {
        self.ssa.reset();
        self.asm.reset(self.asm.reg_limit());
    }

    pub fn vars(&self) -> Arc<BTreeMap<String, u32>> {
        self.ssa.vars.clone()
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
        self.ssa.choice_count
    }

    /// Performs register allocation on a [`ssa::Tape`](SsaTape), building a
    /// complete [`TapeData`](Self).
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let asm = ssa.get_asm(reg_limit);
        Self { ssa, asm }
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    pub fn var_count(&self) -> usize {
        self.ssa.vars.len()
    }

    /// Returns the register limit of the VM tape
    pub fn reg_limit(&self) -> u8 {
        self.asm.reg_limit()
    }

    /// Simplifies both inner tapes, using the provided choice array
    ///
    /// To minimize allocations, this function takes a [`Workspace`](Workspace)
    /// _and_ spare [`TapeData`](TapeData); it will reuse those allocations.
    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        mut tape: TapeData,
    ) -> Result<Self, Error> {
        if choices.len() != self.choice_count() {
            return Err(Error::BadChoiceSlice(
                choices.len(),
                self.choice_count(),
            ));
        }
        let reg_limit = self.asm.reg_limit();
        tape.reset();

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        workspace.reset_with_storage(reg_limit, self.ssa.tape.len(), tape.asm);

        let mut count = 0..;
        let mut choice_count = 0;

        // The tape is constructed so that the output slot is first
        workspace.active[self.ssa.data[0] as usize] =
            Some(count.next().unwrap());

        // Other iterators to consume various arrays in order
        let mut data = self.ssa.data.iter();
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = tape.ssa.tape;
        let mut data_out = tape.ssa.data;

        for &op in self.ssa.tape.iter() {
            let index = *data.next().unwrap();
            if workspace.active[index as usize].is_none() {
                for _ in 0..op.data_count() {
                    data.next().unwrap();
                }
                for _ in 0..op.choice_count() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Because we reassign nodes when they're used as an *input*
            // (while walking the tape in reverse), this node must have been
            // assigned already.
            let new_index = workspace.active[index as usize].unwrap();

            match op {
                SsaOp::Input | SsaOp::CopyImm | SsaOp::Var => {
                    let i = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(i);
                    ops_out.push(op);

                    match op {
                        SsaOp::Input => workspace
                            .alloc
                            .op_input(new_index, i.try_into().unwrap()),
                        SsaOp::Var => workspace.alloc.op_var(new_index, i),
                        SsaOp::CopyImm => workspace
                            .alloc
                            .op_copy_imm(new_index, f32::from_bits(i)),
                        _ => unreachable!(),
                    }
                }
                SsaOp::NegReg
                | SsaOp::AbsReg
                | SsaOp::RecipReg
                | SsaOp::SqrtReg
                | SsaOp::SquareReg => {
                    let arg = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(arg);
                    ops_out.push(op);

                    workspace.alloc.op_reg(new_index, arg, op);
                }
                SsaOp::CopyReg => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    let src = *data.next().unwrap();
                    match workspace.active[src as usize] {
                        Some(new_src) => {
                            data_out.push(new_index);
                            data_out.push(new_src);
                            ops_out.push(op);

                            workspace.alloc.op_reg(
                                new_index,
                                new_src,
                                SsaOp::CopyReg,
                            );
                        }
                        None => {
                            workspace.active[src as usize] = Some(new_index);
                        }
                    }
                }
                SsaOp::MinRegImm | SsaOp::MaxRegImm => {
                    let arg = *data.next().unwrap();
                    let imm = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[arg as usize] {
                            Some(new_arg) => {
                                data_out.push(new_index);
                                data_out.push(new_arg);
                                ops_out.push(SsaOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_arg,
                                    SsaOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[arg as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Right => {
                            data_out.push(new_index);
                            data_out.push(imm);
                            ops_out.push(SsaOp::CopyImm);

                            workspace
                                .alloc
                                .op_copy_imm(new_index, f32::from_bits(imm));
                        }
                        Choice::Both => {
                            choice_count += 1;
                            let arg = *workspace.active[arg as usize]
                                .get_or_insert_with(|| count.next().unwrap());

                            data_out.push(new_index);
                            data_out.push(arg);
                            data_out.push(imm);
                            ops_out.push(op);

                            workspace.alloc.op_reg_imm(
                                new_index,
                                arg,
                                f32::from_bits(imm),
                                op,
                            );
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                SsaOp::MinRegReg | SsaOp::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[lhs as usize] {
                            Some(new_lhs) => {
                                data_out.push(new_index);
                                data_out.push(new_lhs);
                                ops_out.push(SsaOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_lhs,
                                    SsaOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[lhs as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Right => match workspace.active[rhs as usize] {
                            Some(new_rhs) => {
                                data_out.push(new_index);
                                data_out.push(new_rhs);
                                ops_out.push(SsaOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_rhs,
                                    SsaOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[rhs as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            let lhs = *workspace.active[lhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            let rhs = *workspace.active[rhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            data_out.push(new_index);
                            data_out.push(lhs);
                            data_out.push(rhs);
                            ops_out.push(op);

                            workspace.alloc.op_reg_reg(new_index, lhs, rhs, op);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                SsaOp::AddRegReg
                | SsaOp::MulRegReg
                | SsaOp::SubRegReg
                | SsaOp::DivRegReg => {
                    let lhs = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let rhs = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(lhs);
                    data_out.push(rhs);
                    ops_out.push(op);

                    workspace.alloc.op_reg_reg(new_index, lhs, rhs, op);
                }
                SsaOp::AddRegImm
                | SsaOp::MulRegImm
                | SsaOp::SubRegImm
                | SsaOp::SubImmReg
                | SsaOp::DivRegImm
                | SsaOp::DivImmReg => {
                    let arg = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let imm = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(arg);
                    data_out.push(imm);
                    ops_out.push(op);

                    workspace.alloc.op_reg_imm(
                        new_index,
                        arg,
                        f32::from_bits(imm),
                        op,
                    );
                }
            }
        }

        assert_eq!(count.next().unwrap() as usize, ops_out.len());
        let asm_tape = workspace.alloc.finalize();
        assert!(ops_out.len() <= asm_tape.len());

        Ok(TapeData {
            ssa: SsaTape {
                tape: ops_out,
                data: data_out,
                choice_count,
                vars: self.ssa.vars.clone(),
            },
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
        self.ssa.pretty_print()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Data structures used during [`Tape::simplify`]
///
/// This is exposed to minimize reallocations in hot loops.
pub struct Workspace {
    pub alloc: RegisterAllocator,
    pub active: Vec<Option<u32>>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            alloc: RegisterAllocator::empty(),
            active: vec![],
        }
    }
}

impl Workspace {
    /// Resets the workspace, preserving allocations
    pub fn reset(&mut self, num_registers: u8, tape_len: usize) {
        self.alloc.reset(num_registers, tape_len);
        self.active.fill(None);
        self.active.resize(tape_len, None);
    }

    /// Resets the workspace, preserving allocations and claiming the given
    /// [`vm::Tape`](crate::vm::Tape).
    pub fn reset_with_storage(
        &mut self,
        num_registers: u8,
        tape_len: usize,
        tape: VmTape,
    ) {
        self.alloc.reset_with_storage(num_registers, tape_len, tape);
        self.active.fill(None);
        self.active.resize(tape_len, None);
    }
}
