use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};

use bimap::BiMap;
use ordered_float::OrderedFloat;

/// Represents an index in the `Context::nodes` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Node(usize);

/// Represents an index in the `Context::vars` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarNode(usize);

/// Represents an index in the `Context::consts` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConstNode(usize);

#[derive(Debug, Hash, Eq, PartialEq)]
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

#[derive(Debug, Default)]
pub struct Context {
    ops: BiMap<Op, Node>,
    vars: BiMap<String, VarNode>,
    consts: BiMap<OrderedFloat<f64>, ConstNode>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    /// Looks up the [VarNode] corresponding to the given string, inserting
    /// it if it doesn't already exist.
    fn get_var(&mut self, s: &str) -> VarNode {
        let s = s.to_owned();
        match self.vars.get_by_left(&s) {
            Some(v) => *v,
            None => {
                let v = VarNode(self.vars.len());
                self.vars.insert(s, v);
                v
            }
        }
    }
    /// Looks up the [ConstNode] corresponding to the given value, inserting
    /// it if it doesn't already exist.
    fn get_const(&mut self, s: f64) -> ConstNode {
        let s = OrderedFloat(s);
        match self.consts.get_by_left(&s) {
            Some(v) => *v,
            None => {
                let v = ConstNode(self.vars.len());
                self.consts.insert(s, v);
                v
            }
        }
    }
    /// Looks up the [Node] corresponding to the given operation, inserting
    /// it if it doesn't already exist.
    fn get_node(&mut self, op: Op) -> Node {
        match self.ops.get_by_left(&op) {
            Some(v) => *v,
            None => {
                let v = Node(self.ops.len());
                self.ops.insert(op, v);
                v
            }
        }
    }
    /// Looks up the constant associated with the given node. If the node is
    /// invalid for this tree or is not `Op::Const`, returns `None`.
    fn const_value(&self, n: Node) -> Option<f64> {
        match self.ops.get_by_right(&n)? {
            Op::Const(c) => Some(self.consts.get_by_right(c)?.0),
            _ => None,
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Primitives
    pub fn x(&mut self) -> Node {
        let v = self.get_var("X");
        self.get_node(Op::Var(v))
    }
    pub fn y(&mut self) -> Node {
        let v = self.get_var("Y");
        self.get_node(Op::Var(v))
    }
    pub fn z(&mut self) -> Node {
        let v = self.get_var("Z");
        self.get_node(Op::Var(v))
    }
    pub fn constant(&mut self, f: f64) -> Node {
        let v = self.get_const(f);
        self.get_node(Op::Const(v))
    }

    ////////////////////////////////////////////////////////////////////////////
    // Binary functions
    //
    // All of these functions are commutative, which means deduplication is
    // tricky: (a + b) should deduplicate to the same node as (b + a).
    //
    // We put a small amount of effort in to deduplicate: identity operations
    // are collapsed, and non-identity operations are sorted.  There are edge
    // cases that will dodge deduplication: (a + b) + (c + d) will not be
    // deduplicated with (a + c) + (b + d), despite being logically identical.
    pub fn add(&mut self, a: Node, b: Node) -> Node {
        match (self.const_value(a), self.const_value(b)) {
            (Some(a), Some(b)) => self.constant(a + b),
            (Some(zero), _) if zero == 0.0 => b,
            (_, Some(zero)) if zero == 0.0 => a,
            _ if a <= b => self.get_node(Op::Add(a, b)),
            _ => self.get_node(Op::Add(b, a)),
        }
    }
    pub fn mul(&mut self, a: Node, b: Node) -> Node {
        match (self.const_value(a), self.const_value(b)) {
            (Some(a), Some(b)) => self.constant(a * b),
            (Some(one), _) if one == 1.0 => b,
            (_, Some(one)) if one == 1.0 => a,
            _ if a <= b => self.get_node(Op::Mul(a, b)),
            _ => self.get_node(Op::Mul(b, a)),
        }
    }
    pub fn min(&mut self, a: Node, b: Node) -> Node {
        match (self.const_value(a), self.const_value(b)) {
            (Some(a), Some(b)) => self.constant(a.min(b)),
            _ if a == b => a,
            _ if a < b => self.get_node(Op::Min(a, b)),
            _ => self.get_node(Op::Min(b, a)),
        }
    }
    pub fn max(&mut self, a: Node, b: Node) -> Node {
        match (self.const_value(a), self.const_value(b)) {
            (Some(a), Some(b)) => self.constant(a.max(b)),
            _ if a == b => a,
            _ if a < b => self.get_node(Op::Max(a, b)),
            _ => self.get_node(Op::Max(b, a)),
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Unary functions
    fn op_unary<F, G>(&mut self, a: Node, f: F, g: G) -> Node
    where
        F: Fn(f64) -> f64,
        G: Fn(Node) -> Op,
    {
        match self.const_value(a) {
            Some(a) => self.constant(f(a)),
            None => self.get_node(g(a)),
        }
    }
    pub fn neg(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| -a, Op::Neg)
    }
    pub fn recip(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| 1.0 / a, Op::Recip)
    }
    pub fn abs(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.abs(), Op::Abs)
    }
    pub fn sqrt(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.sqrt(), Op::Sqrt)
    }
    pub fn sin(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.sin(), Op::Sin)
    }
    pub fn cos(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.cos(), Op::Cos)
    }
    pub fn tan(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.tan(), Op::Tan)
    }
    pub fn asin(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.asin(), Op::Asin)
    }
    pub fn acos(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.acos(), Op::Acos)
    }
    pub fn atan(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.atan(), Op::Atan)
    }
    pub fn exp(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.exp(), Op::Exp)
    }
    pub fn ln(&mut self, a: Node) -> Node {
        self.op_unary(a, |a| a.ln(), Op::Ln)
    }

    ////////////////////////////////////////////////////////////////////////////
    // Derived functions
    pub fn square(&mut self, a: Node) -> Node {
        self.mul(a, a)
    }
    pub fn sub(&mut self, a: Node, b: Node) -> Node {
        let b = self.neg(b);
        self.add(a, b)
    }
    pub fn div(&mut self, a: Node, b: Node) -> Node {
        let b = self.recip(b);
        self.mul(a, b)
    }

    ////////////////////////////////////////////////////////////////////////////
    pub fn eval(&self, root: Node, vars: &BTreeMap<String, f64>) -> f64 {
        let mut cache = vec![None; self.ops.len()];
        self.eval_inner(root, vars, &mut cache)
    }
    pub fn eval_inner(
        &self,
        node: Node,
        vars: &BTreeMap<String, f64>,
        cache: &mut [Option<f64>],
    ) -> f64 {
        if let Some(v) = cache[node.0] {
            return v;
        }
        let mut get = |n: Node| -> f64 { self.eval_inner(n, vars, cache) };
        let v = match self.ops.get_by_right(&node).unwrap() {
            Op::Var(v) => {
                let var_name = self.vars.get_by_right(v).unwrap();
                *vars.get(var_name).unwrap()
            }
            Op::Const(c) => self.consts.get_by_right(c).unwrap().0,

            Op::Add(a, b) => get(*a) + get(*b),
            Op::Mul(a, b) => get(*a) * get(*b),
            Op::Min(a, b) => get(*a).min(get(*b)),
            Op::Max(a, b) => get(*a).max(get(*b)),

            // Unary operations
            Op::Neg(a) => -get(*a),
            Op::Abs(a) => get(*a).abs(),
            Op::Recip(a) => 1.0 / get(*a),

            // Transcendental functions
            Op::Sqrt(a) => get(*a).sqrt(),
            Op::Sin(a) => get(*a).sin(),
            Op::Cos(a) => get(*a).cos(),
            Op::Tan(a) => get(*a).tan(),
            Op::Asin(a) => get(*a).asin(),
            Op::Acos(a) => get(*a).acos(),
            Op::Atan(a) => get(*a).atan(),
            Op::Exp(a) => get(*a).exp(),
            Op::Ln(a) => get(*a).ln(),
        };

        cache[node.0] = Some(v);
        v
    }

    ////////////////////////////////////////////////////////////////////////////

    /// Parses a flat text representation of a math tree. For example, the
    /// circle (- (+ (square x) (square y)) 1) can be represented as
    /// ```text
    /// 0x600000b90000 var-x
    /// 0x600000b900a0 square 0x600000b90000
    /// 0x600000b90050 var-y
    /// 0x600000b900f0 square 0x600000b90050
    /// 0x600000b90140 add 0x600000b900a0 0x600000b900f0
    /// 0x600000b90190 sqrt 0x600000b90140
    /// 0x600000b901e0 const 1
    /// ```
    ///
    /// This representation is loosely defined and only intended for use in
    /// quick experiments.
    pub fn from_text<R: Read>(r: &mut R) -> (Self, Node) {
        use parse_int::parse;

        let reader = BufReader::new(r);
        let mut ctx = Self::new();
        let mut seen = BTreeMap::new();
        let mut last = None;

        for line in reader.lines().map(|line| line.unwrap()) {
            let mut iter = line.split_whitespace();
            let i: u64 = parse(iter.next().unwrap()).unwrap();
            let opcode = iter.next().unwrap();

            let mut pop = || seen[&parse(iter.next().unwrap()).unwrap()];
            let node = match opcode {
                "const" => ctx.constant(iter.next().unwrap().parse().unwrap()),
                "var-x" => ctx.x(),
                "var-y" => ctx.y(),
                "var-z" => ctx.z(),
                "abs" => ctx.abs(pop()),
                "neg" => ctx.neg(pop()),
                "sqrt" => ctx.sqrt(pop()),
                "square" => ctx.square(pop()),
                "add" => ctx.add(pop(), pop()),
                "mul" => ctx.mul(pop(), pop()),
                "min" => ctx.min(pop(), pop()),
                "max" => ctx.max(pop(), pop()),
                "div" => ctx.div(pop(), pop()),
                "sub" => ctx.sub(pop(), pop()),
                op => panic!("Unknown opcode '{}'", op),
            };
            seen.insert(i, node);
            last = Some(node);
        }
        (ctx, last.unwrap())
    }

    /// Converts the given tree into a GraphViz drawing
    pub fn to_dot<W: Write>(
        &self,
        w: &mut W,
        _root: Node,
    ) -> std::io::Result<()> {
        writeln!(w, "digraph mygraph {{")?;
        for (op, node) in &self.ops {
            // Write node label
            write!(w, r#"n{} [label = ""#, node.0)?;
            match op {
                Op::Const(c) => {
                    let v = self.consts.get_by_right(c).unwrap();
                    write!(w, "{}", v)
                }
                Op::Var(v) => {
                    let v = self.vars.get_by_right(v).unwrap();
                    write!(w, "{}", v)
                }
                Op::Add(..) => write!(w, "add"),
                Op::Mul(..) => write!(w, "mul"),
                Op::Min(..) => write!(w, "min"),
                Op::Max(..) => write!(w, "max"),
                Op::Neg(..) => write!(w, "neg"),
                Op::Abs(..) => write!(w, "abs"),
                Op::Recip(..) => write!(w, "recip"),
                Op::Sqrt(..) => write!(w, "sqrt"),
                Op::Sin(..) => write!(w, "sin"),
                Op::Cos(..) => write!(w, "cos"),
                Op::Tan(..) => write!(w, "tan"),
                Op::Asin(..) => write!(w, "asin"),
                Op::Acos(..) => write!(w, "acos"),
                Op::Atan(..) => write!(w, "atan"),
                Op::Exp(..) => write!(w, "exp"),
                Op::Ln(..) => write!(w, "ln"),
            }?;
            let (color, shape) = match op {
                Op::Const(..) => ("green", "oval"),
                Op::Var(..) => ("red", "circle"),
                Op::Min(..) | Op::Max(..) => ("dodgerblue", "box"),
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
                | Op::Ln(..) => ("goldenrod", "box"),
            };
            let edge_color = format!("{}4", color);
            writeln!(
                w,
                r#"" color="{}1" shape="{}" fontcolor="{}"]"#,
                color, shape, edge_color
            )?;

            match op {
                Op::Add(a, b)
                | Op::Mul(a, b)
                | Op::Min(a, b)
                | Op::Max(a, b) => {
                    writeln!(
                        w,
                        "n{0} -> n{1} [color=\"{3}\"];\
                         n{0} -> n{2} [color=\"{3}\"]",
                        node.0, a.0, b.0, edge_color,
                    )
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
                | Op::Ln(a) => writeln!(
                    w,
                    r#"n{0} -> n{1} [color="{2}"]"#,
                    node.0, a.0, edge_color
                ),

                Op::Var(..) | Op::Const(..) => Ok(()),
            }?;
        }
        writeln!(w, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut ctx = Context::new();
        let x1 = ctx.x();
        let x2 = ctx.x();
        assert_eq!(x1, x2);

        let a = ctx.constant(1.0);
        let b = ctx.constant(1.0);
        assert_eq!(a, b);
        assert_eq!(ctx.const_value(a), Some(1.0));
        assert_eq!(ctx.const_value(x1), None);

        let c = ctx.add(a, b);
        assert_eq!(ctx.const_value(c), Some(2.0));
    }

    #[test]
    fn test_from_text() {
        let txt = "\
0x600000b90000 var-x
0x600000b900a0 square 0x600000b90000
0x600000b90050 var-y
0x600000b900f0 square 0x600000b90050
0x600000b90140 add 0x600000b900a0 0x600000b900f0
0x600000b90190 sqrt 0x600000b90140
0x600000b901e0 const 1";
        let (ctx, _node) = Context::from_text(&mut txt.as_bytes());
        assert_eq!(ctx.len(), 7);
    }
}
