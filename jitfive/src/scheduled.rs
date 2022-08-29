use crate::{
    compiler::{Compiler, NodeIndex, Op, VarIndex},
    indexed::{IndexMap, IndexVec},
};

/// Represents a set of instructions that have been scheduled (somehow)
pub struct Scheduled {
    /// Topologically sorted instruction list, i.e. all nodes are guaranteed to
    /// execute _after_ their inputs.
    pub tape: Vec<(NodeIndex, Op)>,
    pub last_use: IndexVec<usize, NodeIndex>,
    pub vars: IndexMap<String, VarIndex>,
    pub root: NodeIndex,
}

impl Scheduled {
    pub fn new(
        tape: Vec<(NodeIndex, Op)>,
        vars: IndexMap<String, VarIndex>,
        root: NodeIndex,
    ) -> Self {
        let last_use = Self::find_lifetime(&tape);
        Self {
            tape,
            vars,
            root,
            last_use,
        }
    }
    pub fn new_from_compiler(t: &Compiler) -> Self {
        let tape = t
            .flatten()
            .into_iter()
            .map(|n| (n, t.ops[n]))
            .collect::<Vec<_>>();
        let last_use = Self::find_lifetime(&tape);
        Self {
            tape,
            last_use,
            root: t.root,
            vars: t.vars.clone(),
        }
    }
    fn find_lifetime(tape: &[(NodeIndex, Op)]) -> IndexVec<usize, NodeIndex> {
        let max_index = *tape.iter().map(|(n, _op)| n).max().unwrap();
        let mut last_use: IndexVec<usize, NodeIndex> = IndexVec::new();
        last_use.resize(usize::from(max_index) + 1, 0);
        for (i, (_n, op)) in tape.iter().enumerate() {
            match op {
                Op::Var(..) | Op::Const(..) => (),
                Op::Binary(_op, a, b) => {
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Op::BinaryChoice(_op, a, b, ..) => {
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Op::Unary(_op, a) => {
                    last_use[*a] = i;
                }
            }
        }
        last_use
    }
}
