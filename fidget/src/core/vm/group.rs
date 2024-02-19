#![allow(unused)] // TODO remove this
use crate::{
    compiler::{RegTape, RegisterAllocator, SsaOp},
    vm::Choice,
    Error,
};
use std::{collections::HashMap, sync::Arc};

struct GroupData {
    /// Operations in this group, in reverse-evaluation order
    ops: Vec<SsaOp>,

    /// Offset of this group's first choice in a global choice array
    choice_offset: usize,

    /// Subsequent groups which are always enabled if this group is active
    enable_always: Vec<usize>,

    /// Per-group choice data, in reverse-evaluation order
    choices: Vec<ChoiceData>,
}

struct ChoiceData {
    /// Index of the operation to be patched in the `Vec<SsaOp>`
    op_index: usize,

    /// Group to enable if the choice includes `Choice::Left`
    enable_left: usize,

    /// Group to enable if the choice includes `Choice::Right`
    enable_right: usize,
}

struct GroupRoot {
    /// Individual groups, in reverse-evaluation order
    groups: Vec<GroupData>,

    /// Total number of (SSA) operations in all the tape groups
    num_ops: usize,

    /// Number of choice operations in the tape
    choice_count: usize,

    /// Mapping from variable names (in the original [`Context`]) to indexes in
    /// the variable array used during evaluation.
    ///
    /// This is an `Arc` so it can be trivially shared by all of the tape's
    /// descendents, since the variable array order does not change.
    vars: Arc<HashMap<String, u32>>,
}

struct GroupTape {
    /// Handle to the root, containing SSA groups
    root: Arc<GroupRoot>,

    /// Choices made that led to this point
    ///
    /// This array is absolutely indexed, i.e. it has a slot for every choice in
    /// the root tape (though the shortened tape may be using fewer)
    choices: Vec<Choice>,

    /// Active groups, in reverse-evaluation order
    active_groups: Vec<usize>,

    /// Resulting register-allocated tape
    tape: RegTape,
}

struct GroupWorkspace {
    /// Register allocator
    pub(crate) alloc: RegisterAllocator,

    /// Array indicating whether the given group is active
    pub(crate) active: Vec<bool>,
}

impl GroupTape {
    fn simplify(
        &self,
        trace: &[Choice],
        workspace: &mut GroupWorkspace,
        mut out: GroupTape,
    ) -> Result<Self, Error> {
        // TODO check choice count (requires storing it)

        // Mark the root group of this tape as active
        workspace.active.resize(self.root.groups.len(), false);
        workspace.active.fill(false);
        workspace.active[self.active_groups[0]] = true;

        // The new choices array starts out as identical to ours
        let mut choices_out = std::mem::take(&mut out.choices);
        choices_out.resize(self.choices.len(), Choice::Unknown);
        choices_out.copy_from_slice(&self.choices);

        // TODO: plumb register count here
        workspace.alloc.reset(255, self.root.num_ops, out.tape);

        // The new active groups array starts out empty
        let mut groups_out = out.active_groups;
        groups_out.clear();

        let mut trace_iter = trace.iter();
        for &g in &self.active_groups {
            let group = &self.root.groups[g];

            // Skip trace choices if the group is not active
            if !workspace.active[g] {
                for _ in 0..group.choices.len() {
                    trace_iter.next().unwrap();
                }
                continue;
            }
            groups_out.push(g);

            // Enable the always-active nodes connected to this group
            for &e in &group.enable_always {
                workspace.active[e] = true;
            }

            // Prepare to modify the output choice array (which is global) by
            // slicing an appropriate chunk of it.
            let choice_out_slice =
                &mut choices_out[group.choice_offset..][..group.choices.len()];

            let mut choice_meta_iter = group.choices.iter().enumerate();
            for &op in &group.ops {
                let op = if op.has_choice() {
                    // Update the output choice array.  The trace is **bits to
                    // clear**, so we clear them here with bitwise operations.
                    let c = *trace_iter.next().unwrap();
                    let (i, meta) = choice_meta_iter.next().unwrap();
                    choice_out_slice[i] &= !c;

                    // Enable conditional groups and patch the opcode if a
                    // choice was made here
                    match choice_out_slice[i] {
                        Choice::Both => {
                            workspace.active[meta.enable_left] = true;
                            workspace.active[meta.enable_right] = true;
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
                        Choice::Unknown => panic!("cannot plan unknown"),
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
            active_groups: groups_out,
            tape: workspace.alloc.finalize(),
        })
    }
}
