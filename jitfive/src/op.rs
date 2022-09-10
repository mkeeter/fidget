use std::fmt::Write;

use crate::util::indexed::IndexMap;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum UnaryOpcode {
    Neg,
    Abs,
    Recip,
    Sqrt,
    Square,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum BinaryOpcode {
    Add,
    Mul,
    Sub,
    Min,
    Max,
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
pub enum GenericOp<V, F, N> {
    Var(V),
    Const(F),
    Binary(BinaryOpcode, N, N),
    Unary(UnaryOpcode, N),
}

fn dot_color_to_rgb(s: &str) -> &'static str {
    match s {
        "red" => "#FF0000",
        "green" => "#00FF00",
        "goldenrod" => "#DAA520",
        "dodgerblue" => "#1E90FF",
        s => panic!("Unknown X11 color '{}'", s),
    }
}

impl<V, F, N: Copy> GenericOp<V, F, N> {
    pub fn dot_node_color(&self) -> &str {
        match self {
            GenericOp::Const(..) => "green",
            GenericOp::Var(..) => "red",
            GenericOp::Binary(BinaryOpcode::Min | BinaryOpcode::Max, ..) => {
                "dodgerblue"
            }
            GenericOp::Binary(..) | GenericOp::Unary(..) => "goldenrod",
        }
    }
    pub fn dot_node_shape(&self) -> &str {
        match self {
            GenericOp::Const(..) => "oval",
            GenericOp::Var(..) => "circle",
            GenericOp::Binary(..) | GenericOp::Unary(..) => "box",
        }
    }

    pub fn iter_children(&self) -> impl Iterator<Item = N> {
        let out = match self {
            GenericOp::Binary(_, a, b) => [Some(*a), Some(*b)],
            GenericOp::Unary(_, a) => [Some(*a), None],
            GenericOp::Var(..) | GenericOp::Const(..) => [None, None],
        };
        out.into_iter().flatten()
    }
}

impl<V, F, N: Copy> GenericOp<V, F, N>
where
    usize: From<N> + From<V>,
    V: Eq + std::hash::Hash + Copy + From<usize>,
    F: std::fmt::Display,
{
    pub fn dot_node(&self, i: N, vars: &IndexMap<String, V>) -> String {
        let mut out = format!(r#"n{} [label = ""#, usize::from(i));
        match self {
            GenericOp::Const(c) => write!(out, "{}", c).unwrap(),
            GenericOp::Var(v) => {
                let v = vars.get_by_index(*v).unwrap();
                out += v;
            }
            GenericOp::Binary(op, ..) => match op {
                BinaryOpcode::Add => out += "add",
                BinaryOpcode::Mul => out += "mul",
                BinaryOpcode::Sub => out += "sub",
                BinaryOpcode::Min => out += "min",
                BinaryOpcode::Max => out += "max",
            },
            GenericOp::Unary(op, ..) => match op {
                UnaryOpcode::Neg => out += "neg",
                UnaryOpcode::Abs => out += "abs",
                UnaryOpcode::Recip => out += "recip",
                UnaryOpcode::Sqrt => out += "sqrt",
                UnaryOpcode::Square => out += "square",
            },
        };
        write!(
            out,
            r#"" color="{0}1" shape="{1}" fontcolor="{0}4"]"#,
            self.dot_node_color(),
            self.dot_node_shape()
        )
        .unwrap();
        out
    }

    pub fn dot_edges(&self, i: N) -> String {
        let mut out = String::new();
        for c in self.iter_children() {
            out += &self.dot_edge(i, c, "FF");
        }
        out
    }

    pub fn dot_edge(&self, a: N, b: N, alpha: &str) -> String {
        let color = dot_color_to_rgb(self.dot_node_color()).to_owned() + alpha;
        format!(
            "n{} -> n{} [color = \"{color}\"]\n",
            usize::from(a),
            usize::from(b),
        )
    }
}
