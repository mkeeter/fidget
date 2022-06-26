use crate::indexed::define_index;

define_index!(Node, "An index in the `Context::nodes` map");
define_index!(VarNode, "An index in the `Context::vars` map");
define_index!(ConstNode, "An index in the `Context::consts` map");

impl Node {
    pub fn dot_name(&self) -> String {
        format!("n{}", self.0)
    }
}

/// Represents an operation in a math expression.
///
/// An `Op` is tightly coupled to the [`Context`](crate::context::Context)
/// which generated it, and will not be valid for a different `Context`.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum Op {
    Var(VarNode),
    Const(ConstNode),

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
}
