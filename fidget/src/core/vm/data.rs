//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    compiler::{RegOp, RegTape, RegisterAllocator, SsaOp, SsaTape},
    context::{Context, Node},
    vm::Choice,
    Error,
};
use std::{collections::HashMap, sync::Arc};

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
/// use fidget::{compiler::RegOp, eval::MathShape, vm::{VmShape, VmData}};
///
/// let (sum, ctx) = fidget::rhai::eval("x + y")?;
/// let data = VmData::<255>::new(&ctx, sum)?;
/// assert_eq!(data.len(), 3); // X, Y, and (X + Y)
///
/// let mut iter = data.iter_asm();
/// assert_eq!(iter.next().unwrap(), RegOp::Input(0, 0));
/// assert_eq!(iter.next().unwrap(), RegOp::Input(1, 1));
/// assert_eq!(iter.next().unwrap(), RegOp::AddRegReg(0, 0, 1));
/// # Ok::<(), fidget::Error>(())
/// ```
///
#[derive(Default)]
pub struct VmData<const N: usize = { u8::MAX as usize }> {
    asm: RegTape,

    /// Number of choice operations in the tape
    pub choice_count: usize,

    /// Mapping from variable names (in the original [`Context`]) to indexes in
    /// the variable array used during evaluation.
    ///
    /// This is an `Arc` so it can be trivially shared by all of the tape's
    /// descendents, since the variable array order does not change.
    pub vars: Arc<HashMap<String, u32>>,
}

impl<const N: usize> VmData<N> {
    /// Builds a new tape for the given node
    pub fn new(context: &Context, node: Node) -> Result<Self, Error> {
        let ssa = SsaTape::new(context, node)?;
        let asm = RegTape::new::<N>(&ssa);
        Ok(Self {
            asm,
            choice_count: ssa.choice_count,
            vars: ssa.vars,
        })
    }

    /// Returns this tape's mapping of variable names to indexes
    pub fn vars(&self) -> &HashMap<String, u32> {
        &self.vars
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
        self.choice_count
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.asm.slot_count()
    }

    /// Returns the number of variables used in this tape
    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    /// Simplifies both inner tapes, using the provided choice array
    ///
    /// To minimize allocations, this function takes a [`VmWorkspace`] and
    /// spare [`VmData`]; it will reuse those allocations.
    pub fn simplify(
        &self,
        choices: &[Choice],
        workspace: &mut VmWorkspace<N>,
        tape: VmData<N>,
    ) -> Result<Self, Error> {
        if choices.len() != self.choice_count() {
            return Err(Error::BadChoiceSlice(
                choices.len(),
                self.choice_count(),
            ));
        }

        // Steal `tape.asm` and hand it to the workspace for use in allocator
        // TODO this overestimates because it includes load/store
        workspace.reset(self.asm.len(), tape.asm);

        let mut choice_count = 0;

        // The tape is constructed so that the output slot is first
        workspace.set_active(0, 0);
        workspace.count += 1;

        // Other iterators to consume various arrays in order
        let mut choice_iter = choices.iter().rev();

        // TODO use borrowed array for RegTape::ssa() workspace
        for mut op in self.asm.ssa() {
            let index = op.output();

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
        }

        //assert_eq!(workspace.count as usize, ops_out.len());
        let asm_tape = workspace.alloc.finalize();

        Ok(VmData {
            choice_count,
            asm: asm_tape,
            vars: self.vars.clone(),
        })
    }

    /// Produces an iterator that visits [`RegOp`] values in evaluation order
    pub fn iter_asm(&self) -> impl Iterator<Item = RegOp> + '_ {
        self.asm.iter().cloned().rev()
    }

    /// Pretty-prints the inner SSA tape
    pub fn pretty_print(&self) {
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
