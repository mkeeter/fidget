use crate::indexed::define_index;
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
}
