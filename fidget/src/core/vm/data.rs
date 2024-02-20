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

        // Register 0 must be the first SSA slot
        workspace.reg(0);

        // Other iterators to consume various arrays in order
        let mut choice_iter = choices.iter().rev();

        for op in self.asm.iter() {
            // Skip clauses which are inactive, but handle their output binding
            // and choice in the choice iterator.
            let index = op.output();
            if !workspace.active(index) {
                // manually clear the binding
                workspace.bind[index as usize] = u32::MAX;
                if op.has_choice() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Convert from RegOp -> SsaOp
            let op = match *op {
                RegOp::Load(reg, mem) => {
                    let prev = workspace.out(reg);
                    assert_eq!(workspace.bind[mem as usize], u32::MAX);
                    workspace.bind[mem as usize] = prev;
                    continue;
                }
                RegOp::Store(reg, mem) => {
                    let prev = workspace.bind[mem as usize];
                    workspace.bind[mem as usize] = u32::MAX;
                    assert_eq!(workspace.bind[reg as usize], u32::MAX);
                    workspace.bind[reg as usize] = prev;
                    continue;
                }
                RegOp::Input(out, i) => {
                    SsaOp::Input(workspace.out(out), i as u32)
                }
                RegOp::Var(out, i) => SsaOp::Var(workspace.out(out), i),
                RegOp::NegReg(out, arg) => {
                    SsaOp::NegReg(workspace.out(out), workspace.reg(arg))
                }
                RegOp::AbsReg(out, arg) => {
                    SsaOp::AbsReg(workspace.out(out), workspace.reg(arg))
                }
                RegOp::RecipReg(out, arg) => {
                    SsaOp::RecipReg(workspace.out(out), workspace.reg(arg))
                }
                RegOp::SqrtReg(out, arg) => {
                    SsaOp::SqrtReg(workspace.out(out), workspace.reg(arg))
                }
                RegOp::CopyReg(out, arg) => {
                    let Some(op) = workspace.copy(out, arg) else {
                        continue;
                    };
                    op
                }
                RegOp::SquareReg(out, arg) => {
                    SsaOp::SquareReg(workspace.out(out), workspace.reg(arg))
                }
                RegOp::AddRegReg(out, lhs, rhs) => SsaOp::AddRegReg(
                    workspace.out(out),
                    workspace.reg(lhs),
                    workspace.reg(rhs),
                ),
                RegOp::MulRegReg(out, lhs, rhs) => SsaOp::MulRegReg(
                    workspace.out(out),
                    workspace.reg(lhs),
                    workspace.reg(rhs),
                ),
                RegOp::DivRegReg(out, lhs, rhs) => SsaOp::DivRegReg(
                    workspace.out(out),
                    workspace.reg(lhs),
                    workspace.reg(rhs),
                ),
                RegOp::SubRegReg(out, lhs, rhs) => SsaOp::SubRegReg(
                    workspace.out(out),
                    workspace.reg(lhs),
                    workspace.reg(rhs),
                ),
                RegOp::MinRegReg(out, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            let Some(op) = workspace.copy(out, lhs) else {
                                continue;
                            };
                            op
                        }
                        Choice::Right => {
                            let Some(op) = workspace.copy(out, rhs) else {
                                continue;
                            };
                            op
                        }
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MinRegReg(
                                workspace.out(out),
                                workspace.reg(lhs),
                                workspace.reg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            let Some(op) = workspace.copy(out, lhs) else {
                                continue;
                            };
                            op
                        }
                        Choice::Right => {
                            let Some(op) = workspace.copy(out, rhs) else {
                                continue;
                            };
                            op
                        }
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MaxRegReg(
                                workspace.out(out),
                                workspace.reg(lhs),
                                workspace.reg(rhs),
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                RegOp::AddRegImm(out, arg, imm) => SsaOp::AddRegImm(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::MulRegImm(out, arg, imm) => SsaOp::MulRegImm(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::DivRegImm(out, arg, imm) => SsaOp::DivRegImm(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::DivImmReg(out, arg, imm) => SsaOp::DivImmReg(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::SubImmReg(out, arg, imm) => SsaOp::SubImmReg(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::SubRegImm(out, arg, imm) => SsaOp::SubRegImm(
                    workspace.out(out),
                    workspace.reg(arg),
                    imm,
                ),
                RegOp::MinRegImm(out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            let Some(op) = workspace.copy(out, arg) else {
                                continue;
                            };
                            op
                        }
                        Choice::Right => {
                            SsaOp::CopyImm(workspace.out(out), imm)
                        }
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MinRegImm(
                                workspace.out(out),
                                workspace.reg(arg),
                                imm,
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => {
                            let Some(op) = workspace.copy(out, arg) else {
                                continue;
                            };
                            op
                        }
                        Choice::Right => {
                            SsaOp::CopyImm(workspace.out(out), imm)
                        }
                        Choice::Both => {
                            choice_count += 1;
                            SsaOp::MaxRegImm(
                                workspace.out(out),
                                workspace.reg(arg),
                                imm,
                            )
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                RegOp::CopyImm(out, imm) => {
                    SsaOp::CopyImm(workspace.out(out), imm)
                }
            };

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
    /// Returns a register -> SSA binding, assigning a new binding if missing
    fn reg(&mut self, i: u8) -> u32 {
        if self.bind[i as usize] == u32::MAX {
            self.bind[i as usize] = self.count;
            self.count += 1;
        }
        self.bind[i as usize]
    }

    /// Returns a register -> SSA binding, clearing that binding upon return
    fn out(&mut self, i: u8) -> u32 {
        assert_ne!(self.bind[i as usize], u32::MAX);
        let out = self.bind[i as usize];
        self.bind[i as usize] = u32::MAX;
        out
    }

    /// Checks whether the given register is active, clearing it if so
    fn active(&self, v: u32) -> bool {
        self.bind[v as usize] != u32::MAX
    }

    fn copy(&mut self, dst: u8, src: u8) -> Option<SsaOp> {
        let dst = self.out(dst);
        if self.bind[src as usize] != u32::MAX {
            Some(SsaOp::CopyReg(dst, self.reg(src)))
        } else {
            assert_eq!(self.bind[src as usize], u32::MAX);
            self.bind[src as usize] = dst;
            None
        }
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
