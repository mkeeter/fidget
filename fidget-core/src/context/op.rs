use crate::{
    context::{Node, indexed::Index},
    types::FloatExt,
    var::Var,
};
use ordered_float::OrderedFloat;

/// A one-argument math operation
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum UnaryOpcode {
    Neg,
    Abs,
    Recip,
    Sqrt,
    Square,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Ln,
    Not,
    Rand,
}

/// A two-argument math operation
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum BinaryOpcode {
    Add,
    Sub,
    Mul,
    Div,
    Atan,
    Min,
    Max,
    Compare,
    Mod,
    And,
    Or,
    Mix,
}

impl BinaryOpcode {
    /// Evaluates the opcode
    pub fn eval(&self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOpcode::Add => a + b,
            BinaryOpcode::Sub => a - b,
            BinaryOpcode::Mul => a * b,
            BinaryOpcode::Div => a / b,
            BinaryOpcode::Atan => a.atan2(b),
            BinaryOpcode::Min => a.min_choice(b).0,
            BinaryOpcode::Max => a.max_choice(b).0,
            BinaryOpcode::Compare => a.compare(b),
            BinaryOpcode::Mod => a.rem_euclid(b),
            BinaryOpcode::And => a.and_choice(b).0,
            BinaryOpcode::Or => a.or_choice(b).0,
            BinaryOpcode::Mix => a.mix(b),
        }
    }
}

impl UnaryOpcode {
    /// Evaluates the opcode
    pub fn eval(&self, a: f32) -> f32 {
        match self {
            UnaryOpcode::Neg => -a,
            UnaryOpcode::Abs => a.abs(),
            UnaryOpcode::Recip => 1.0 / a,
            UnaryOpcode::Sqrt => a.sqrt(),
            UnaryOpcode::Square => a * a,
            UnaryOpcode::Floor => a.floor(),
            UnaryOpcode::Ceil => a.ceil(),
            UnaryOpcode::Round => a.round(),
            UnaryOpcode::Sin => a.sin(),
            UnaryOpcode::Cos => a.cos(),
            UnaryOpcode::Tan => a.tan(),
            UnaryOpcode::Asin => a.asin(),
            UnaryOpcode::Acos => a.acos(),
            UnaryOpcode::Atan => a.atan(),
            UnaryOpcode::Exp => a.exp(),
            UnaryOpcode::Ln => a.ln(),
            UnaryOpcode::Not => a.not(),
            UnaryOpcode::Rand => a.rand(),
        }
    }
}

/// An operation in a math expression
///
/// `Op`s should be constructed by calling functions on
/// [`Context`](crate::context::Context), e.g.
/// [`Context::add`](crate::context::Context::add) will generate an
/// `Op::Binary(BinaryOpcode::Add, .., ..)` node and return an opaque handle.
///
/// Each `Op` is tightly coupled to the [`Context`](crate::context::Context)
/// which generated it, and will not be valid for a different `Context`.
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum Op {
    Input(Var),
    Const(OrderedFloat<f32>),
    Binary(BinaryOpcode, Node, Node),
    Unary(UnaryOpcode, Node),
}

fn dot_color_to_rgb(s: &str) -> &'static str {
    match s {
        "red" => "#FF0000",
        "green" => "#00FF00",
        "goldenrod" => "#DAA520",
        "dodgerblue" => "#1E90FF",
        s => panic!("Unknown X11 color '{s}'"),
    }
}

impl Op {
    /// Returns the color to be used in a GraphViz drawing for this node
    pub fn dot_node_color(&self) -> &str {
        match self {
            Op::Const(..) => "green",
            Op::Input(..) => "red",
            Op::Binary(BinaryOpcode::Min | BinaryOpcode::Max, ..) => {
                "dodgerblue"
            }
            Op::Binary(..) | Op::Unary(..) => "goldenrod",
        }
    }

    /// Returns the shape to be used in a GraphViz drawing for this node
    pub fn dot_node_shape(&self) -> &str {
        match self {
            Op::Const(..) => "oval",
            Op::Input(..) => "circle",
            Op::Binary(..) | Op::Unary(..) => "box",
        }
    }

    /// Iterates over children, producing 0, 1, or 2 values
    pub fn iter_children(&self) -> impl Iterator<Item = Node> {
        let out = match self {
            Op::Binary(_, a, b) => [Some(*a), Some(*b)],
            Op::Unary(_, a) => [Some(*a), None],
            Op::Input(..) | Op::Const(..) => [None, None],
        };
        out.into_iter().flatten()
    }

    /// Returns a GraphViz string of edges from this node to its children
    pub fn dot_edges(&self, i: Node) -> String {
        let mut out = String::new();
        for c in self.iter_children() {
            out += &self.dot_edge(i, c, "FF");
        }
        out
    }

    /// Returns a single edge with user-specified transparency
    pub fn dot_edge(&self, a: Node, b: Node, alpha: &str) -> String {
        let color = dot_color_to_rgb(self.dot_node_color()).to_owned() + alpha;
        format!("n{} -> n{} [color = \"{color}\"]\n", a.get(), b.get(),)
    }
}
