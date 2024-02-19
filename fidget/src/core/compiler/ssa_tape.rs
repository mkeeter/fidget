//use crate::vm::{RegisterAllocator, Tape as VmTape};
use crate::{
    compiler::SsaOp,
    context::{BinaryOpcode, Node, Op, UnaryOpcode},
    Context, Error,
};

use std::collections::{HashMap, HashSet};

/// Instruction tape, storing groups of [opcodes in SSA form](crate::compiler::SsaOp)
///
/// The [`SsaRoot`] is typically stored in an `Arc`, so all of its fields are
/// public (but should be immutable in normal usage).
#[derive(Debug)]
pub struct SsaRoot {
    /// Individual groups, in reverse-evaluation order
    pub groups: Vec<SsaGroup>,

    /// Total number of (SSA) operations in all the tape groups
    pub num_ops: usize,

    /// Number of choice operations in the tape
    pub choice_count: usize,

    /// Mapping from variable names (in the original [`Context`]) to indexes in
    /// the variable array used during evaluation.
    ///
    /// This is an `Arc` so it can be trivially shared by all of the tape's
    /// descendents, since the variable array order does not change.
    pub vars: HashMap<String, u32>,
}

/// Individual group of [SSA operations](SsaOp)
#[derive(Debug)]
pub struct SsaGroup {
    /// Operations in this group, in reverse-evaluation order
    pub ops: Vec<SsaOp>,

    /// Offset of this group's first choice in a global choice array
    pub choice_offset: usize,

    /// Subsequent groups which are **always** enabled if this group is active
    pub enable_always: Vec<usize>,

    /// Per-group choice data, in reverse-evaluation order
    pub choices: Vec<SsaChoiceData>,
}

/// Downstream groups to enable for a given choice
#[derive(Debug)]
pub struct SsaChoiceData {
    /// Group to enable if the choice includes `Choice::Left`
    pub enable_left: usize,

    /// Group to enable if the choice includes `Choice::Right`
    pub enable_right: usize,
}

impl SsaRoot {
    /// Flattens a subtree of the graph into groups of straight-line code.
    ///
    /// This should always succeed unless the `root` is from a different
    /// `Context`, in which case `Error::BadNode` will be returned.
    pub fn new(ctx: &Context, root: Node) -> Result<Self, Error> {
        let mut mapping = HashMap::new();
        let mut parent_count: HashMap<Node, usize> = HashMap::new();
        let mut var_names = HashMap::new();
        let mut slot_count = 0;

        // Get either a node or constant index
        #[derive(Copy, Clone)]
        enum Slot {
            Reg(u32),
            Immediate(f32),
        }

        // Accumulate parent counts and declare all nodes
        let mut seen = HashSet::new();
        let mut todo = vec![root];
        while let Some(node) = todo.pop() {
            if !seen.insert(node) {
                continue;
            }
            let op = ctx.get_op(node).ok_or(Error::BadNode)?;
            let prev = match op {
                Op::Const(c) => {
                    mapping.insert(node, Slot::Immediate(c.0 as f32))
                }
                _ => {
                    let i = slot_count;
                    slot_count += 1;
                    if matches!(op, Op::Var(..)) {
                        let next_var = var_names.len().try_into().unwrap();
                        var_names.insert(
                            ctx.var_name(node).unwrap().unwrap().to_string(),
                            next_var,
                        );
                    }
                    mapping.insert(node, Slot::Reg(i))
                }
            };
            assert!(prev.is_none());
            for child in op.iter_children() {
                *parent_count.entry(child).or_default() += 1;
                todo.push(child);
            }
        }

        // Now that we've populated our parents, flatten the graph
        let mut seen = HashSet::new();
        let mut todo = vec![root];
        let mut choice_count = 0;
        let mut tape = vec![];
        while let Some(node) = todo.pop() {
            if *parent_count.get(&node).unwrap_or(&0) > 0 || !seen.insert(node)
            {
                continue;
            }
            let op = ctx.get_op(node).unwrap();
            for child in op.iter_children() {
                todo.push(child);
                *parent_count.get_mut(&child).unwrap() -= 1;
            }

            let Slot::Reg(i) = mapping[&node] else {
                // Constants are skipped, because they become immediates
                continue;
            };
            let op = match op {
                Op::Input(..) => {
                    let arg = match ctx.var_name(node).unwrap().unwrap() {
                        "X" => 0,
                        "Y" => 1,
                        "Z" => 2,
                        i => panic!("Unexpected input index: {i}"),
                    };
                    SsaOp::Input(i, arg)
                }
                Op::Var(..) => {
                    let v = ctx.var_name(node).unwrap().unwrap();
                    let arg = var_names[v];
                    SsaOp::Var(i, arg)
                }
                Op::Const(..) => {
                    unreachable!("skipped above")
                }
                Op::Binary(op, lhs, rhs) => {
                    let lhs = mapping[lhs];
                    let rhs = mapping[rhs];

                    type RegFn = fn(u32, u32, u32) -> SsaOp;
                    type ImmFn = fn(u32, u32, f32) -> SsaOp;
                    let f: (RegFn, ImmFn, ImmFn) = match op {
                        BinaryOpcode::Add => (
                            SsaOp::AddRegReg,
                            SsaOp::AddRegImm,
                            SsaOp::AddRegImm,
                        ),
                        BinaryOpcode::Sub => (
                            SsaOp::SubRegReg,
                            SsaOp::SubRegImm,
                            SsaOp::SubImmReg,
                        ),
                        BinaryOpcode::Mul => (
                            SsaOp::MulRegReg,
                            SsaOp::MulRegImm,
                            SsaOp::MulRegImm,
                        ),
                        BinaryOpcode::Div => (
                            SsaOp::DivRegReg,
                            SsaOp::DivRegImm,
                            SsaOp::DivImmReg,
                        ),
                        BinaryOpcode::Min => (
                            SsaOp::MinRegReg,
                            SsaOp::MinRegImm,
                            SsaOp::MinRegImm,
                        ),
                        BinaryOpcode::Max => (
                            SsaOp::MaxRegReg,
                            SsaOp::MaxRegImm,
                            SsaOp::MaxRegImm,
                        ),
                    };

                    if matches!(op, BinaryOpcode::Min | BinaryOpcode::Max) {
                        choice_count += 1;
                    }

                    match (lhs, rhs) {
                        (Slot::Reg(lhs), Slot::Reg(rhs)) => f.0(i, lhs, rhs),
                        (Slot::Reg(arg), Slot::Immediate(imm)) => {
                            f.1(i, arg, imm)
                        }
                        (Slot::Immediate(imm), Slot::Reg(arg)) => {
                            f.2(i, arg, imm)
                        }
                        (Slot::Immediate(..), Slot::Immediate(..)) => {
                            panic!("Cannot handle f(imm, imm)")
                        }
                    }
                }
                Op::Unary(op, lhs) => {
                    let lhs = match mapping[lhs] {
                        Slot::Reg(r) => r,
                        Slot::Immediate(..) => {
                            panic!("Cannot handle f(imm)")
                        }
                    };
                    let op = match op {
                        UnaryOpcode::Neg => SsaOp::NegReg,
                        UnaryOpcode::Abs => SsaOp::AbsReg,
                        UnaryOpcode::Recip => SsaOp::RecipReg,
                        UnaryOpcode::Sqrt => SsaOp::SqrtReg,
                        UnaryOpcode::Square => SsaOp::SquareReg,
                    };
                    op(i, lhs)
                }
            };
            tape.push(op);
        }

        // Special case if the Node is a single constant, which isn't usually
        // recorded in the tape
        if tape.is_empty() {
            let c = ctx.const_value(root).unwrap().unwrap() as f32;
            tape.push(SsaOp::CopyImm(0, c));
        }

        Ok(SsaRoot {
            groups: todo!(),
            num_ops: todo!(),
            choice_count,
            vars: var_names,
        })
    }

    /// Returns the total number of opcodes
    pub fn len(&self) -> usize {
        self.num_ops
    }

    /// Pretty-prints the given tape to `stdout`
    pub fn pretty_print(&self) {
        for &op in self.groups.iter().rev().flat_map(|g| g.ops.iter().rev()) {
            match op {
                SsaOp::Input(out, i) => {
                    println!("${out} = INPUT {i}");
                }
                SsaOp::Var(out, i) => {
                    println!("${out} = VAR {i}");
                }
                SsaOp::NegReg(out, arg)
                | SsaOp::AbsReg(out, arg)
                | SsaOp::RecipReg(out, arg)
                | SsaOp::SqrtReg(out, arg)
                | SsaOp::CopyReg(out, arg)
                | SsaOp::SquareReg(out, arg) => {
                    let op = match op {
                        SsaOp::NegReg(..) => "NEG",
                        SsaOp::AbsReg(..) => "ABS",
                        SsaOp::RecipReg(..) => "RECIP",
                        SsaOp::SqrtReg(..) => "SQRT",
                        SsaOp::SquareReg(..) => "SQUARE",
                        SsaOp::CopyReg(..) => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${arg}");
                }

                SsaOp::AddRegReg(out, lhs, rhs)
                | SsaOp::MulRegReg(out, lhs, rhs)
                | SsaOp::DivRegReg(out, lhs, rhs)
                | SsaOp::SubRegReg(out, lhs, rhs)
                | SsaOp::MinRegReg(out, lhs, rhs)
                | SsaOp::MaxRegReg(out, lhs, rhs) => {
                    let op = match op {
                        SsaOp::AddRegReg(..) => "ADD",
                        SsaOp::MulRegReg(..) => "MUL",
                        SsaOp::DivRegReg(..) => "DIV",
                        SsaOp::SubRegReg(..) => "SUB",
                        SsaOp::MinRegReg(..) => "MIN",
                        SsaOp::MaxRegReg(..) => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                SsaOp::AddRegImm(out, arg, imm)
                | SsaOp::MulRegImm(out, arg, imm)
                | SsaOp::DivRegImm(out, arg, imm)
                | SsaOp::DivImmReg(out, arg, imm)
                | SsaOp::SubImmReg(out, arg, imm)
                | SsaOp::SubRegImm(out, arg, imm)
                | SsaOp::MinRegImm(out, arg, imm)
                | SsaOp::MaxRegImm(out, arg, imm) => {
                    let (op, swap) = match op {
                        SsaOp::AddRegImm(..) => ("ADD", false),
                        SsaOp::MulRegImm(..) => ("MUL", false),
                        SsaOp::DivImmReg(..) => ("DIV", true),
                        SsaOp::DivRegImm(..) => ("DIV", false),
                        SsaOp::SubImmReg(..) => ("SUB", true),
                        SsaOp::SubRegImm(..) => ("SUB", false),
                        SsaOp::MinRegImm(..) => ("MIN", false),
                        SsaOp::MaxRegImm(..) => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                SsaOp::CopyImm(out, imm) => {
                    println!("${out} = COPY {imm}");
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ring() {
        let mut ctx = Context::new();
        let c0 = ctx.constant(0.5);
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r = ctx.add(x2, y2).unwrap();
        let c6 = ctx.sub(r, c0).unwrap();
        let c7 = ctx.constant(0.25);
        let c8 = ctx.sub(c7, r).unwrap();
        let c9 = ctx.max(c8, c6).unwrap();

        let root = SsaRoot::new(&ctx, c9).unwrap();
        assert_eq!(root.len(), 8);
    }

    #[test]
    fn test_dupe() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let x_squared = ctx.mul(x, x).unwrap();

        let root = SsaRoot::new(&ctx, x_squared).unwrap();
        assert_eq!(root.len(), 2);
    }
}
