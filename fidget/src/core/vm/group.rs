#![allow(unused)] // TODO remove this
use crate::{
    compiler::{RegTape, SsaOp},
    vm::Choice,
};
use std::sync::Arc;

struct GroupData {
    /// Operations in this group, in reverse-evaluation order
    ops: Vec<SsaOp>,

    /// Offset of this group's first choice in a global choice array
    choice_offset: usize,

    /// Subsequent groups which are always enabled if this group is active
    enable_always: Vec<usize>,

    /// Choice data, in reverse-evaluation order
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

struct GroupTape {
    root: Arc<Vec<GroupData>>,
    choice: Vec<Choice>,
    active_groups: Vec<usize>,
    tape: RegTape,
}

impl GroupTape {
    fn simplify(&self, trace: &[Choice]) -> Self {
        // TODO: could this only include groups in this particular tape?
        let mut active = vec![false; self.root.len()];

        // New choice array
        let mut choices_out = self.choice.clone();

        // Mark the root group of this tape as active
        active[self.active_groups[0]] = true;

        let mut choice_iter = trace.iter();

        for &g in &self.active_groups {
            if !active[g] {
                continue;
            }
            let group = &self.root[g];
            for &e in &group.enable_always {
                active[e] = true;
            }
            let choice_out_slice =
                &mut choices_out[group.choice_offset..][..group.choices.len()];

            let mut choice_meta_iter = group.choices.iter().enumerate();
            for op in &self.root[g].ops {
                if op.has_choice() {
                    let c = *choice_iter.next().unwrap();
                    let (i, meta) = choice_meta_iter.next().unwrap();
                    choice_out_slice[i] &= !c;
                    // TODO: generate appropriate RegOp
                } else {
                    // TODO: generate appropriate RegOp
                }
            }
            assert!(choice_meta_iter.next().is_none());
        }
        assert!(choice_iter.next().is_none());
        todo!()
    }
}
