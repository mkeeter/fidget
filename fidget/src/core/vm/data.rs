//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    compiler::{RegOp, RegRegAlloc, RegTape, SsaTape},
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
        workspace.reset(self.asm.slot_count(), tape.asm);

        let mut choice_count = 0;

        // Register 0 must be the first SSA slot
        workspace.set_active(0);

        // Other iterators to consume various arrays in order
        let mut choice_iter = choices.iter().rev();

        for op in self.asm.iter() {
            // Skip clauses which are inactive, but handle their output binding
            // and choice in the choice iterator.
            let index = op.output();
            if !workspace.take_active(index) {
                if op.has_choice() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Convert from RegOp -> SsaOp
            let op = match *op {
                RegOp::MinRegReg(out, lhs, rhs)
                | RegOp::MaxRegReg(out, lhs, rhs) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => RegOp::CopyReg(out, lhs),
                        Choice::Right => RegOp::CopyReg(out, rhs),
                        Choice::Both => {
                            choice_count += 1;
                            *op
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                RegOp::MinRegImm(out, arg, imm)
                | RegOp::MaxRegImm(out, arg, imm) => {
                    match choice_iter.next().unwrap() {
                        Choice::Left => RegOp::CopyReg(out, arg),
                        Choice::Right => RegOp::CopyImm(out, imm),
                        Choice::Both => {
                            choice_count += 1;
                            *op
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                op => op,
            };
            for c in op.iter_children() {
                workspace.set_active(c);
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
    pub(crate) alloc: RegRegAlloc<N>,

    /// Array indicating which registers (or memory slots) are active
    pub(crate) active: Vec<bool>,
}

impl<const N: usize> Default for VmWorkspace<N> {
    fn default() -> Self {
        Self {
            alloc: RegRegAlloc::empty(),
            active: vec![],
        }
    }
}

impl<const N: usize> VmWorkspace<N> {
    /// Sets the given register as active
    fn set_active(&mut self, v: u32) {
        self.active[v as usize] = true
    }

    /// Checks whether the given register is active, clearing it
    fn take_active(&mut self, v: u32) -> bool {
        std::mem::take(&mut self.active[v as usize])
    }

    /// Resets the workspace, preserving allocations and claiming the given
    /// [`RegTape`].
    pub fn reset(&mut self, slot_count: usize, tape: RegTape) {
        self.alloc.reset(slot_count, tape);
        self.active.resize(slot_count, false);
        self.active.fill(false);
    }
}
