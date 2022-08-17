use std::io::Write;

use crate::{
    error::Error,
    indexed::{define_index, IndexMap},
};

use ordered_float::OrderedFloat;

define_index!(Node, "An index in the `Context::ops` map");
define_index!(VarNode, "An index in the `Context::vars` map");

impl Node {
    pub fn dot_name(&self) -> String {
        format!("n{}", self.0)
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum UnaryOpcode {
    Neg,
    Abs,
    Recip,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Ln,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum BinaryOpcode {
    Add,
    Mul,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum BinaryChoiceOpcode {
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
pub enum GenericOp<V, F, N, C = ()> {
    Var(V),
    Const(F),
    Binary(BinaryOpcode, N, N),
    BinaryChoice(BinaryChoiceOpcode, N, N, C),
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

impl<V, F, N: Copy, C> GenericOp<V, F, N, C> {
    pub fn dot_node_color(&self) -> &str {
        match self {
            GenericOp::Const(..) => "green",
            GenericOp::Var(..) => "red",
            GenericOp::BinaryChoice(..) => "dodgerblue",
            GenericOp::Binary(..) | GenericOp::Unary(..) => "goldenrod",
        }
    }
    pub fn dot_node_shape(&self) -> &str {
        match self {
            GenericOp::Const(..) => "oval",
            GenericOp::Var(..) => "circle",
            GenericOp::Binary(..)
            | GenericOp::Unary(..)
            | GenericOp::BinaryChoice(..) => "box",
        }
    }

    pub fn iter_children(&self) -> impl Iterator<Item = N> {
        let out = match self {
            GenericOp::Binary(_, a, b)
            | GenericOp::BinaryChoice(_, a, b, _) => [Some(*a), Some(*b)],
            GenericOp::Unary(_, a) => [Some(*a), None],
            GenericOp::Var(..) | GenericOp::Const(..) => [None, None],
        };
        out.into_iter().flatten()
    }
}

impl<V, F, N: Copy, C> GenericOp<V, F, N, C>
where
    usize: From<N> + From<V>,
    V: Eq + std::hash::Hash + Copy + From<usize>,
    F: std::fmt::Display,
{
    pub fn write_dot<W: Write>(
        &self,
        w: &mut W,
        i: N,
        vars: &IndexMap<String, V>,
    ) -> Result<(), Error> {
        write!(w, r#"n{} [label = ""#, usize::from(i))?;
        match self {
            GenericOp::Const(c) => {
                write!(w, "{}", c)
            }
            GenericOp::Var(v) => {
                let v = vars.get_by_index(*v).ok_or(Error::BadVar)?;
                write!(w, "{}", v)
            }
            GenericOp::Binary(op, ..) => match op {
                BinaryOpcode::Add => write!(w, "add"),
                BinaryOpcode::Mul => write!(w, "mul"),
            },
            GenericOp::BinaryChoice(op, ..) => match op {
                BinaryChoiceOpcode::Min => write!(w, "min"),
                BinaryChoiceOpcode::Max => write!(w, "max"),
            },
            GenericOp::Unary(op, ..) => match op {
                UnaryOpcode::Neg => write!(w, "neg"),
                UnaryOpcode::Abs => write!(w, "abs"),
                UnaryOpcode::Recip => write!(w, "recip"),
                UnaryOpcode::Sqrt => write!(w, "sqrt"),
                UnaryOpcode::Sin => write!(w, "sin"),
                UnaryOpcode::Cos => write!(w, "cos"),
                UnaryOpcode::Tan => write!(w, "tan"),
                UnaryOpcode::Asin => write!(w, "asin"),
                UnaryOpcode::Acos => write!(w, "acos"),
                UnaryOpcode::Atan => write!(w, "atan"),
                UnaryOpcode::Exp => write!(w, "exp"),
                UnaryOpcode::Ln => write!(w, "ln"),
            },
        }?;
        writeln!(
            w,
            r#"" color="{0}1" shape="{1}" fontcolor="{0}4"]"#,
            self.dot_node_color(),
            self.dot_node_shape()
        )?;
        Ok(())
    }

    pub fn write_dot_edge<W: Write>(
        &self,
        w: &mut W,
        a: N,
        b: N,
        alpha: &str,
    ) -> Result<(), Error> {
        let color = dot_color_to_rgb(self.dot_node_color()).to_owned() + alpha;
        writeln!(
            w,
            "n{} -> n{} [color = \"{color}\"]",
            usize::from(a),
            usize::from(b),
        )?;
        Ok(())
    }
    pub fn write_dot_edges<W: Write>(
        &self,
        w: &mut W,
        i: N,
    ) -> Result<(), Error> {
        for c in self.iter_children() {
            self.write_dot_edge(w, i, c, "FF")?;
        }
        Ok(())
    }
    pub fn write_dot_edges_invis<W: Write>(
        &self,
        w: &mut W,
        i: N,
    ) -> Result<(), Error> {
        for c in self.iter_children() {
            writeln!(
                w,
                "n{} -> n{} [style=invis]",
                usize::from(i),
                usize::from(c),
            )?;
        }
        Ok(())
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
