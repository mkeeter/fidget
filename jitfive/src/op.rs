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

/// Represents an operation in a math expression.
///
/// `Op`s should be constructed by calling functions on
/// [`Context`](crate::context::Context), e.g.
/// [`Context::add`](crate::context::Context::add) will generate an `Op::Add`
/// node and return an opaque handle.
///
/// Each `Op` is tightly coupled to the [`Context`](crate::context::Context)
/// which generated it, and will not be valid for a different `Context`.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum Op {
    Var(VarNode),
    Const(OrderedFloat<f64>),

    // Commutative ops
    Add(Node, Node),
    Mul(Node, Node),
    Min(Node, Node),
    Max(Node, Node),

    // Unary operations
    Neg(Node),
    Abs(Node),
    Recip(Node),

    // Transcendental functions
    Sqrt(Node),
    Sin(Node),
    Cos(Node),
    Tan(Node),
    Asin(Node),
    Acos(Node),
    Atan(Node),
    Exp(Node),
    Ln(Node),
}

impl Op {
    pub fn dot_node_color(&self) -> &str {
        match self {
            Op::Const(..) => "green",
            Op::Var(..) => "red",
            Op::Min(..) | Op::Max(..) => "dodgerblue",
            Op::Add(..)
            | Op::Mul(..)
            | Op::Neg(..)
            | Op::Abs(..)
            | Op::Recip(..)
            | Op::Sqrt(..)
            | Op::Sin(..)
            | Op::Cos(..)
            | Op::Tan(..)
            | Op::Asin(..)
            | Op::Acos(..)
            | Op::Atan(..)
            | Op::Exp(..)
            | Op::Ln(..) => "goldenrod",
        }
    }
    pub fn dot_node_shape(&self) -> &str {
        match self {
            Op::Const(..) => "oval",
            Op::Var(..) => "circle",
            Op::Min(..) | Op::Max(..) => "box",
            Op::Add(..)
            | Op::Mul(..)
            | Op::Neg(..)
            | Op::Abs(..)
            | Op::Recip(..)
            | Op::Sqrt(..)
            | Op::Sin(..)
            | Op::Cos(..)
            | Op::Tan(..)
            | Op::Asin(..)
            | Op::Acos(..)
            | Op::Atan(..)
            | Op::Exp(..)
            | Op::Ln(..) => "box",
        }
    }

    pub fn iter_children(&self) -> impl Iterator<Item = Node> {
        let out = match self {
            Op::Min(a, b) | Op::Max(a, b) | Op::Add(a, b) | Op::Mul(a, b) => {
                [Some(*a), Some(*b)]
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
            | Op::Ln(a) => [Some(*a), None],

            Op::Var(..) | Op::Const(..) => [None, None],
        };
        out.into_iter().flatten()
    }

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
            Op::Add(a, b) | Op::Mul(a, b) | Op::Min(a, b) | Op::Max(a, b) => {
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
