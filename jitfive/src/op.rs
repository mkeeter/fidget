use crate::{
    indexed::{define_index, IndexMap},
    program::{ChoiceIndex, Instruction, RegIndex, VarIndex},
};

use ordered_float::OrderedFloat;

define_index!(Node, "An index in the `Context::ops` map");
define_index!(VarNode, "An index in the `Context::vars` map");

impl Node {
    pub fn dot_name(&self) -> String {
        format!("n{}", self.0)
    }
}

/// Represents an generic operation
///
/// Parameterized by four types:
/// - `V` is an index type associated with `Var` nodes
/// - `F` is the type used to store floating-point values
/// - `N` is the index type for inter-op references
/// - `C` is a choice index type attached to each min/max node (which can be
///   empty at certain points in the pipeline)
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum GenericOp<V, F, N, C = ()> {
    Var(V),
    Const(F),

    // Commutative ops
    Add(N, N),
    Mul(N, N),
    Min(N, N, C),
    Max(N, N, C),

    // Unary operations
    Neg(N),
    Abs(N),
    Recip(N),

    // Transcendental functions
    Sqrt(N),
    Sin(N),
    Cos(N),
    Tan(N),
    Asin(N),
    Acos(N),
    Atan(N),
    Exp(N),
    Ln(N),
}

impl<V, F, N: Copy, C> GenericOp<V, F, N, C> {
    pub fn dot_node_color(&self) -> &str {
        match self {
            GenericOp::Const(..) => "green",
            GenericOp::Var(..) => "red",
            GenericOp::Min(..) | GenericOp::Max(..) => "dodgerblue",
            GenericOp::Add(..)
            | GenericOp::Mul(..)
            | GenericOp::Neg(..)
            | GenericOp::Abs(..)
            | GenericOp::Recip(..)
            | GenericOp::Sqrt(..)
            | GenericOp::Sin(..)
            | GenericOp::Cos(..)
            | GenericOp::Tan(..)
            | GenericOp::Asin(..)
            | GenericOp::Acos(..)
            | GenericOp::Atan(..)
            | GenericOp::Exp(..)
            | GenericOp::Ln(..) => "goldenrod",
        }
    }
    pub fn dot_node_shape(&self) -> &str {
        match self {
            GenericOp::Const(..) => "oval",
            GenericOp::Var(..) => "circle",
            GenericOp::Min(..) | GenericOp::Max(..) => "box",
            GenericOp::Add(..)
            | GenericOp::Mul(..)
            | GenericOp::Neg(..)
            | GenericOp::Abs(..)
            | GenericOp::Recip(..)
            | GenericOp::Sqrt(..)
            | GenericOp::Sin(..)
            | GenericOp::Cos(..)
            | GenericOp::Tan(..)
            | GenericOp::Asin(..)
            | GenericOp::Acos(..)
            | GenericOp::Atan(..)
            | GenericOp::Exp(..)
            | GenericOp::Ln(..) => "box",
        }
    }

    pub fn iter_children(&self) -> impl Iterator<Item = N> {
        use GenericOp as Op;
        let out = match self {
            Op::Min(a, b, _)
            | GenericOp::Max(a, b, _)
            | GenericOp::Add(a, b)
            | GenericOp::Mul(a, b) => [Some(*a), Some(*b)],

            GenericOp::Neg(a)
            | GenericOp::Abs(a)
            | GenericOp::Recip(a)
            | GenericOp::Sqrt(a)
            | GenericOp::Sin(a)
            | GenericOp::Cos(a)
            | GenericOp::Tan(a)
            | GenericOp::Asin(a)
            | GenericOp::Acos(a)
            | GenericOp::Atan(a)
            | GenericOp::Exp(a)
            | GenericOp::Ln(a) => [Some(*a), None],

            GenericOp::Var(..) | GenericOp::Const(..) => [None, None],
        };
        out.into_iter().flatten()
    }
}

/// Represents an operation in a math expression.
///
/// `Op`s should be constructed by calling functions on
/// [`Context`](crate::context::Context), e.g.
/// [`Context::add`](crate::context::Context::add) will generate an `Op::Add`
/// node and return an opaque handle.
///
/// Each `Op` is tightly coupled to the [`Context`](crate::context::Context)
/// which generated it, and will not be valid for a different `Context`.
pub type Op = GenericOp<VarNode, OrderedFloat<f64>, Node>;

impl Op {
    /// Converts the given `Op` into an `Instruction`, freeing it from its
    /// parent context.
    ///
    /// `id` is the `Node` handle for this `Op`.
    ///
    /// This will _always_ assign a register in `regs`. If the node is
    /// `Op::Var` or `Op::Min/Max`, it will assign an index in `vars` or
    /// `choices` respectively.
    pub fn to_instruction(
        &self,
        id: Node,
        regs: &mut IndexMap<Node, RegIndex>,
        vars: &mut IndexMap<VarNode, VarIndex>,
        choices: &mut IndexMap<Node, ChoiceIndex>,
    ) -> Instruction {
        let out = regs.insert(id);

        match self {
            Op::Var(v) => {
                let var = vars.insert(*v);
                Instruction::Var { var, out }
            }
            Op::Const(f) => Instruction::Const { value: f.0, out },

            // Two-argument operations
            Op::Add(a, b)
            | Op::Mul(a, b)
            | Op::Min(a, b, _)
            | Op::Max(a, b, _) => {
                let lhs = regs.insert(*a);
                let rhs = regs.insert(*b);
                match self {
                    Op::Add(..) => Instruction::Add { lhs, rhs, out },
                    Op::Mul(..) => Instruction::Mul { lhs, rhs, out },
                    Op::Min(..) | Op::Max(..) => {
                        let choice = choices.insert(id);
                        match self {
                            Op::Min(..) => Instruction::Min {
                                lhs,
                                rhs,
                                choice,
                                out,
                            },
                            Op::Max(..) => Instruction::Max {
                                lhs,
                                rhs,
                                choice,
                                out,
                            },
                            _ => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                }
            }

            Op::Neg(a)
            | Op::Abs(a)
            | Op::Recip(a)
            | Op::Sqrt(a)
            | Op::Sin(a)
            | Op::Cos(a)
            | Op::Tan(a)
            | Op::Asin(a)
            | Op::Acos(a)
            | Op::Atan(a)
            | Op::Exp(a)
            | Op::Ln(a) => {
                let reg = regs.insert(*a);
                match self {
                    Op::Neg(..) => Instruction::Neg { reg, out },
                    Op::Abs(..) => Instruction::Abs { reg, out },
                    Op::Recip(..) => Instruction::Recip { reg, out },
                    Op::Sqrt(..) => Instruction::Sqrt { reg, out },
                    Op::Sin(..) => Instruction::Sin { reg, out },
                    Op::Cos(..) => Instruction::Cos { reg, out },
                    Op::Tan(..) => Instruction::Tan { reg, out },
                    Op::Asin(..) => Instruction::Asin { reg, out },
                    Op::Acos(..) => Instruction::Acos { reg, out },
                    Op::Atan(..) => Instruction::Atan { reg, out },
                    Op::Exp(..) => Instruction::Exp { reg, out },
                    Op::Ln(..) => Instruction::Ln { reg, out },
                    _ => unreachable!(),
                }
            }
        }
    }
}
