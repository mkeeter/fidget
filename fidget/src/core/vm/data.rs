//! General-purpose tapes for use during evaluation or further compilation
use crate::{
    compiler::{RegOp, RegTape, RegisterAllocator, SsaOp, SsaRoot},
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
#[derive(Debug)]
pub struct VmData<const N: u8 = { u8::MAX }> {
    /// Handle to the root, containing SSA groups
    root: Arc<SsaRoot>,

    /// Choices made that led to this point
    ///
    /// This array is absolutely indexed, i.e. it has a slot for every choice in
    /// the root tape (though the shortened tape may be using fewer)
    choices: Vec<Choice>,

    /// Number of active choices in this shortened tape
    ///
    /// This will always be `<= self.choices.len()`
    choice_count: usize,

    /// Active groups, in reverse-evaluation order
    active_groups: Vec<usize>,

    /// Resulting register-allocated tape
    tape: RegTape,
}

/// Storage associated with `VmData`, to reduce memory allocation
#[derive(Default)]
pub struct VmStorage {
    /// See [`VmData::choices`]
    choices: Vec<Choice>,

    /// See [`VmData::active_groups`]
    active_groups: Vec<usize>,

    /// See [`VmData::tape`]
    tape: RegTape,
}

impl<const N: u8> VmData<N> {
    /// Builds a new tape for the given node
    pub fn new(context: &Context, node: Node) -> Result<Self, Error> {
        let root = SsaRoot::new(context, node)?;
        let tape = RegTape::new(&root, N);
        let choices = vec![Choice::Both; root.choice_count];
        let choice_count = root.choice_count;
        let active_groups = (0..root.groups.len()).collect();
        Ok(Self {
            root: root.into(),
            choices,
            choice_count,
            active_groups,
            tape,
        })
    }

    /// Returns this tape's mapping of variable names to indexes
    pub fn vars(&self) -> &HashMap<String, u32> {
        &self.root.vars
    }

    /// Returns the length of the internal VM tape
    pub fn len(&self) -> usize {
        self.tape.len()
    }

    /// Returns true if the internal VM tape is empty
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }

    /// Returns the number of choice (min/max) nodes in the shortened tape.
    ///
    /// This is required because some evaluators pre-allocate spaces for the
    /// choice array.
    pub fn choice_count(&self) -> usize {
        self.choice_count
    }

    /// Returns the number of slots used by the inner VM tape
    pub fn slot_count(&self) -> usize {
        self.tape.slot_count()
    }

    /// Returns the number of variables used in this tape
    pub fn var_count(&self) -> usize {
        self.root.vars.len()
    }

    /// Simplifies both inner tapes, using the provided choice array
    ///
    /// To minimize allocations, this function takes a [`VmWorkspace`] and
    /// spare [`VmData`]; it will reuse those allocations.
    pub fn simplify(
        &self,
        trace: &[Choice],
        workspace: &mut VmWorkspace,
        mut out: VmStorage,
    ) -> Result<Self, Error> {
        // TODO check choice count (requires storing it)

        // Mark the root group of this tape as active
        workspace.active.resize(self.root.groups.len(), false);
        workspace.active.fill(false);
        workspace.active[self.active_groups[0]] = true;

        // The new choices array starts out as identical to ours
        let mut choices_out = std::mem::take(&mut out.choices);
        choices_out.resize(self.choices.len(), Choice::None);
        choices_out.copy_from_slice(&self.choices);

        workspace.alloc.reset(N, self.root.len(), out.tape);

        // The new active groups array starts out empty
        let mut groups_out = out.active_groups;
        groups_out.clear();

        let mut choice_count = 0;
        let mut trace_iter = trace.iter();
        for &g in &self.active_groups {
            let group = &self.root.groups[g];

            // Prepare to modify the output choice array (which is global) by
            // slicing an appropriate chunk of it.
            let choice_out_slice =
                &mut choices_out[group.choice_offset..][..group.choices.len()];

            // Skip choices in the evaluation trace if the group is not active
            // but the choice was previously active
            if !workspace.active[g] {
                for c in choice_out_slice {
                    if *c == Choice::Both {
                        trace_iter.next().unwrap();
                    }
                }
                continue;
            }

            // Otherwise, we need to simplify this group
            groups_out.push(g);

            // Enable the always-active nodes connected to this group
            for &e in &group.enable_always {
                workspace.active[e] = true;
            }

            let mut choice_meta_iter = group.choices.iter().enumerate();
            for &op in &group.ops {
                let op = if op.has_choice() {
                    // Update the output choice array, if the choice was written
                    // here (i.e. it was Choice::Both in the previous evaluation
                    // trace)
                    let (i, meta) = choice_meta_iter.next().unwrap();
                    if choice_out_slice[i] == Choice::Both {
                        let c = *trace_iter.next().unwrap();
                        choice_out_slice[i] &= c;
                    }

                    // Enable conditional groups and patch the opcode if a
                    // choice was made here
                    match choice_out_slice[i] {
                        Choice::Both => {
                            workspace.active[meta.enable_left] = true;
                            workspace.active[meta.enable_right] = true;
                            choice_count += 1;
                            op
                        }
                        Choice::Left => {
                            workspace.active[meta.enable_left] = true;
                            match op {
                                SsaOp::MinRegReg(out, lhs, ..)
                                | SsaOp::MaxRegReg(out, lhs, ..)
                                | SsaOp::MinRegImm(out, lhs, ..)
                                | SsaOp::MaxRegImm(out, lhs, ..) => {
                                    SsaOp::CopyReg(out, lhs)
                                }
                                _ => panic!("invalid choice op"),
                            }
                        }
                        Choice::Right => {
                            workspace.active[meta.enable_right] = true;
                            match op {
                                SsaOp::MinRegReg(out, _lhs, rhs)
                                | SsaOp::MaxRegReg(out, _lhs, rhs) => {
                                    SsaOp::CopyReg(out, rhs)
                                }
                                SsaOp::MinRegImm(out, _lhs, imm)
                                | SsaOp::MaxRegImm(out, _lhs, imm) => {
                                    SsaOp::CopyImm(out, imm)
                                }
                                _ => panic!("invalid choice op"),
                            }
                        }
                        Choice::None => panic!("cannot plan unknown"),
                    }
                } else {
                    op
                };
                workspace.alloc.op(op);
            }
            assert!(choice_meta_iter.next().is_none());
        }
        assert!(trace_iter.next().is_none());

        Ok(Self {
            root: self.root.clone(),
            choices: choices_out,
            choice_count,
            active_groups: groups_out,
            tape: workspace.alloc.finalize(),
        })
    }

    /// Produces an iterator that visits [`RegOp`] values in evaluation order
    pub fn iter_asm(&self) -> impl Iterator<Item = RegOp> + '_ {
        self.tape.iter().cloned().rev()
    }

    /// Pretty-prints the inner SSA tape
    pub fn pretty_print(&self) {
        self.root.pretty_print();
    }

    /// Recycles memory allocations for later reuse
    pub fn recycle(self) -> VmStorage {
        VmStorage {
            choices: self.choices,
            active_groups: self.active_groups,
            tape: self.tape,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Data structures used during [`VmData::simplify`]
///
/// This is exposed to minimize reallocations in hot loops.
pub struct VmWorkspace {
    /// Register allocator
    pub(crate) alloc: RegisterAllocator,

    /// Array indicating whether the given group is active
    pub(crate) active: Vec<bool>,
}

impl Default for VmWorkspace {
    fn default() -> Self {
        Self {
            alloc: RegisterAllocator::empty(),
            active: vec![],
        }
    }
}
