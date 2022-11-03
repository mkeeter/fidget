use crate::{
    asm::{AsmOp, AsmTape, RegisterAllocator},
    eval::Choice,
    tape::{SsaTape, TapeOp},
};
use std::sync::Arc;

/// Light-weight handle for tape data
///
/// This can be passed by value and cloned.
#[derive(Clone)]
pub struct Tape(Arc<TapeData>);
impl Tape {
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let t = TapeData::from_ssa(ssa, reg_limit);
        Self(Arc::new(t))
    }

    pub fn simplify(&self, choices: &[Choice]) -> Self {
        self.simplify_with(choices, &mut Default::default(), Default::default())
    }

    /// Simplifies a tape, reusing workspace and allocations
    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        prev: TapeData,
    ) -> Self {
        let t = self.0.simplify_with(choices, workspace, prev);
        Self(Arc::new(t))
    }

    pub fn take(mut self) -> Option<TapeData> {
        Arc::get_mut(&mut self.0).map(std::mem::take)
    }
}

impl std::ops::Deref for Tape {
    type Target = TapeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`Tape`](Self) stores two different representations:
/// - A tape in SSA form, suitable for use during tape simplification
/// - A [`Vec<AsmOp>`](crate::asm::AsmOp), ready to be fed into an assembler,
///   (e.g. [`dynasm`](crate::asm::dynasm)).
///
/// We keep both because SSA form makes tape shortening easier, while the `asm`
/// data already has registers assigned for lowering into machine assembly.
#[derive(Default)]
pub struct TapeData {
    ssa: SsaTape,
    asm: AsmTape,
}

impl TapeData {
    pub fn reset(&mut self) {
        self.ssa.reset();
        self.asm.reset(self.asm.reg_limit());
    }
    /// Returns the length of the internal `AsmOp` tape
    pub fn len(&self) -> usize {
        self.asm.len()
    }
    pub fn is_empty(&self) -> bool {
        self.asm.is_empty()
    }

    /// Returns the number of choice (min/max) nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocated spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.ssa.choice_count
    }
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let asm = ssa.get_asm(reg_limit);
        Self { ssa, asm }
    }

    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    pub fn simplify_with(
        &self,
        choices: &[Choice],
        workspace: &mut Workspace,
        mut tape: TapeData,
    ) -> Self {
        let reg_limit = self.asm.reg_limit();
        tape.reset();

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        workspace.reset_give(reg_limit, self.ssa.tape.len(), tape.asm);

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
                TapeOp::Input | TapeOp::CopyImm => {
                    let i = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(i);
                    ops_out.push(op);

                    match op {
                        TapeOp::Input => workspace
                            .alloc
                            .op_input(new_index, i.try_into().unwrap()),
                        TapeOp::CopyImm => workspace
                            .alloc
                            .op_copy_imm(new_index, f32::from_bits(i)),
                        _ => unreachable!(),
                    }
                }
                TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::SquareReg => {
                    let arg = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(arg);
                    ops_out.push(op);

                    workspace.alloc.op_reg(new_index, arg, op);
                }
                TapeOp::CopyReg => {
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
                                TapeOp::CopyReg,
                            );
                        }
                        None => {
                            workspace.active[src as usize] = Some(new_index);
                        }
                    }
                }
                TapeOp::MinRegImm | TapeOp::MaxRegImm => {
                    let arg = *data.next().unwrap();
                    let imm = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[arg as usize] {
                            Some(new_arg) => {
                                data_out.push(new_index);
                                data_out.push(new_arg);
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_arg,
                                    TapeOp::CopyReg,
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
                            ops_out.push(TapeOp::CopyImm);

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
                TapeOp::MinRegReg | TapeOp::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[lhs as usize] {
                            Some(new_lhs) => {
                                data_out.push(new_index);
                                data_out.push(new_lhs);
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_lhs,
                                    TapeOp::CopyReg,
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
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_rhs,
                                    TapeOp::CopyReg,
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
                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::SubRegReg
                | TapeOp::DivRegReg => {
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
                TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::SubRegImm
                | TapeOp::SubImmReg
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg => {
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
        let asm_tape = workspace.alloc.take();
        assert!(ops_out.len() <= asm_tape.len());

        TapeData {
            ssa: SsaTape {
                tape: ops_out,
                data: data_out,
                choice_count,
            },
            asm: asm_tape,
        }
    }

    /// Produces an iterator that visits [`AsmOp`](crate::asm::AsmOp) values in
    /// evaluation order.
    pub fn iter_asm(&self) -> impl Iterator<Item = AsmOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    pub fn pretty_print(&self) {
        self.ssa.pretty_print()
    }
}

////////////////////////////////////////////////////////////////////////////////

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
    pub fn reset(&mut self, num_registers: u8, tape_len: usize) {
        self.alloc.reset(num_registers, tape_len);
        self.active.fill(None);
        self.active.resize(tape_len, None);
    }
    pub fn reset_give(
        &mut self,
        num_registers: u8,
        tape_len: usize,
        tape: AsmTape,
    ) {
        self.alloc.reset_give(num_registers, tape_len, tape);
        self.active.fill(None);
        self.active.resize(tape_len, None);
    }
}
