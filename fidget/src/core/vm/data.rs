//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    bytecode::Bytecode,
    compiler::{RegOp, RegTape, RegisterAllocator, SsaOp, SsaTape},
    context::{Context, Node},
    var::VarMap,
    vm::Choice,
    Error,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A flattened math expression, ready for evaluation or further compilation.
///
/// Under the hood, [`VmData`] stores two different representations:
/// - A tape in [single static assignment form](https://en.wikipedia.org/wiki/Static_single-assignment_form)
///   ([`SsaTape`]), which is suitable for use during tape simplification
/// - A tape in register-allocated form ([`RegTape`]), which can be efficiently
///   evaluated or lowered into machine assembly
///
/// # Example
/// Consider the expression `x + y`.  The SSA tape will look something like
/// this:
/// ```text
/// $0 = INPUT 0   // X
/// $1 = INPUT 1   // Y
/// $2 = ADD $0 $1 // (X + Y)
/// ```
///
/// This will be lowered into a tape using real (or VM) registers:
/// ```text
/// r0 = INPUT 0 // X
/// r1 = INPUT 1 // Y
/// r0 = ADD r0 r1 // (X + Y)
/// ```
///
/// Note that in this form, registers are reused (e.g. `r0` stores both `X` and
/// `X + Y`).
///
/// We can peek at the internals and see this register-allocated tape:
/// ```
/// use fidget::{
///     compiler::RegOp,
///     context::{Context, Tree},
///     vm::VmData,
///     var::Var,
/// };
///
/// let tree = Tree::x() + Tree::y();
/// let mut ctx = Context::new();
/// let sum = ctx.import(&tree);
/// let data = VmData::<255>::new(&ctx, &[sum])?;
/// assert_eq!(data.len(), 4); // X, Y, (X + Y), and output
///
/// let mut iter = data.iter_asm();
/// let vars = &data.vars; // map from var to index
/// assert_eq!(iter.next().unwrap(), RegOp::Input(0, vars[&Var::X] as u32));
/// assert_eq!(iter.next().unwrap(), RegOp::Input(1, vars[&Var::Y] as u32));
/// assert_eq!(iter.next().unwrap(), RegOp::AddRegReg(0, 0, 1));
/// # Ok::<(), fidget::Error>(())
/// ```
///
/// Despite this peek at its internals, users are unlikely to touch `VmData`
/// directly; a [`VmShape`](crate::vm::VmShape) wraps the `VmData` and
/// implements our common traits.
#[derive(Default, Serialize, Deserialize)]
pub struct VmData<const N: usize = { u8::MAX as usize }> {
    ssa: SsaTape,
    asm: RegTape,

    /// Mapping from variables to indices during evaluation
    ///
    /// This member is stored in a shared pointer because it's passed down to
    /// children (constructed with [`VmData::simplify`]).
    pub vars: Arc<VarMap>,
}

impl<const N: usize> VmData<N> {
    /// Builds a new tape for the given node
    pub fn new(context: &Context, nodes: &[Node]) -> Result<Self, Error> {
        let (ssa, vars) = SsaTape::new(context, nodes)?;
        let asm = RegTape::new::<N>(&ssa);
        Ok(Self {
            ssa,
            asm,
            vars: vars.into(),
        })
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

    /// Returns the number of output nodes in the tape.
    ///
    /// This is required because some evaluators pre-allocate spaces for the
    /// output array.
    pub fn output_count(&self) -> usize {
        self.ssa.output_count
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    /// Simplifies both inner tapes, using the provided choice array
    ///
    /// To minimize allocations, this function takes a [`VmWorkspace`] and
    /// spare [`VmData`]; it will reuse those allocations.
    pub fn simplify<const M: usize>(
        &self,
        choices: &[Choice],
        workspace: &mut VmWorkspace<M>,
        mut tape: VmData<M>,
    ) -> Result<VmData<M>, Error> {
        if choices.len() != self.choice_count() {
            return Err(Error::BadChoiceSlice(
                choices.len(),
                self.choice_count(),
            ));
        }
        tape.ssa.reset();

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        workspace.reset(self.ssa.tape.len(), tape.asm);

        let mut choice_count = 0;
        let mut output_count = 0;

        // Other iterators to consume various arrays in order
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = tape.ssa.tape;

        for mut op in self.ssa.tape.iter().cloned() {
            let index = match &mut op {
                SsaOp::Output(reg, _i) => {
                    *reg = workspace.get_or_insert_active(*reg);
                    workspace.alloc.op(op);
                    ops_out.push(op);
                    output_count += 1;
                    continue;
                }
                _ => op.output().unwrap(),
            };

            if workspace.active(index).is_none() {
                if op.has_choice() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Because we reassign nodes when they're used as an *input*
            // (while walking the tape in reverse), this node must have been
            // assigned already.
            let new_index = workspace.active(index).unwrap();

            match &mut op {
                SsaOp::Output(..) => unreachable!(),
                SsaOp::Input(index, ..) | SsaOp::CopyImm(index, ..) => {
                    *index = new_index;
                }
                SsaOp::NegReg(index, arg)
                | SsaOp::AbsReg(index, arg)
                | SsaOp::RecipReg(index, arg)
                | SsaOp::SqrtReg(index, arg)
                | SsaOp::SquareReg(index, arg)
                | SsaOp::FloorReg(index, arg)
                | SsaOp::CeilReg(index, arg)
                | SsaOp::RoundReg(index, arg)
                | SsaOp::SinReg(index, arg)
                | SsaOp::CosReg(index, arg)
                | SsaOp::TanReg(index, arg)
                | SsaOp::AsinReg(index, arg)
                | SsaOp::AcosReg(index, arg)
                | SsaOp::AtanReg(index, arg)
                | SsaOp::ExpReg(index, arg)
                | SsaOp::LnReg(index, arg)
                | SsaOp::NotReg(index, arg) => {
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
                | SsaOp::MaxRegImm(index, arg, imm)
                | SsaOp::AndRegImm(index, arg, imm)
                | SsaOp::OrRegImm(index, arg, imm) => {
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
                | SsaOp::MaxRegReg(index, lhs, rhs)
                | SsaOp::AndRegReg(index, lhs, rhs)
                | SsaOp::OrRegReg(index, lhs, rhs) => {
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
                | SsaOp::DivRegReg(index, lhs, rhs)
                | SsaOp::AtanRegReg(index, lhs, rhs)
                | SsaOp::CompareRegReg(index, lhs, rhs)
                | SsaOp::ModRegReg(index, lhs, rhs) => {
                    *index = new_index;
                    *lhs = workspace.get_or_insert_active(*lhs);
                    *rhs = workspace.get_or_insert_active(*rhs);
                }
                SsaOp::AddRegImm(index, arg, _imm)
                | SsaOp::MulRegImm(index, arg, _imm)
                | SsaOp::SubRegImm(index, arg, _imm)
                | SsaOp::SubImmReg(index, arg, _imm)
                | SsaOp::DivRegImm(index, arg, _imm)
                | SsaOp::DivImmReg(index, arg, _imm)
                | SsaOp::AtanImmReg(index, arg, _imm)
                | SsaOp::AtanRegImm(index, arg, _imm)
                | SsaOp::CompareRegImm(index, arg, _imm)
                | SsaOp::CompareImmReg(index, arg, _imm)
                | SsaOp::ModRegImm(index, arg, _imm)
                | SsaOp::ModImmReg(index, arg, _imm) => {
                    *index = new_index;
                    *arg = workspace.get_or_insert_active(*arg);
                }
            }
            workspace.alloc.op(op);
            ops_out.push(op);
        }

        assert_eq!(workspace.count as usize + 1, ops_out.len());
        let asm_tape = workspace.alloc.finalize();

        Ok(VmData {
            ssa: SsaTape {
                tape: ops_out,
                choice_count,
                output_count,
            },
            asm: asm_tape,
            vars: self.vars.clone(),
        })
    }

    /// Produces an iterator that visits [`RegOp`] values in evaluation order
    pub fn iter_asm(&self) -> impl Iterator<Item = RegOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    /// Converts the inner [RegOp] tape to bytecode
    ///
    /// See [bytecode](crate::bytecode) for details on the format
    pub fn to_bytecode(&self) -> Bytecode {
        if N == u8::MAX as usize {
            self.asm.to_bytecode()
        } else {
            RegTape::new::<{ u8::MAX as usize }>(&self.ssa).to_bytecode()
        }
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

/// Data structures used during [`VmData::simplify`]
///
/// This is exposed to minimize reallocations in hot loops.
pub struct VmWorkspace<const N: usize> {
    /// Register allocator
    pub(crate) alloc: RegisterAllocator<N>,

    /// Current bindings from SSA variables to registers
    pub(crate) bind: Vec<u32>,

    /// Number of active SSA bindings
    ///
    /// This value is monotonically increasing; each SSA variable gets the next
    /// value if it is unassigned when encountered.
    count: u32,
}

impl<const N: usize> Default for VmWorkspace<N> {
    fn default() -> Self {
        Self {
            alloc: RegisterAllocator::empty(),
            bind: vec![],
            count: 0,
        }
    }
}

impl<const N: usize> VmWorkspace<N> {
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

    /// Resets the workspace, preserving allocations and claiming the given
    /// [`RegTape`].
    pub fn reset(&mut self, tape_len: usize, tape: RegTape) {
        self.alloc.reset(tape_len, tape);
        self.bind.fill(u32::MAX);
        self.bind.resize(tape_len, u32::MAX);
        self.count = 0;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simplify_reg_count_change() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();
        let xy = ctx.add(x, y).unwrap();
        let xyz = ctx.add(xy, z).unwrap();

        let data = VmData::<3>::new(&ctx, &[xyz]).unwrap();
        assert_eq!(data.len(), 6); // 3x input, 2x add, 1x output
        let next = data
            .simplify::<2>(&[], &mut Default::default(), Default::default())
            .unwrap();
        assert_eq!(next.len(), 8); // extra load + store

        let data = VmData::<2>::new(&ctx, &[xyz]).unwrap();
        assert_eq!(data.len(), 8);
        let next = data
            .simplify::<3>(&[], &mut Default::default(), Default::default())
            .unwrap();
        assert_eq!(next.len(), 6);
    }
}
