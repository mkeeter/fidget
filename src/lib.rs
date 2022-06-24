use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};

use ordered_float::OrderedFloat;
use thiserror::Error;

mod vecmap;

#[derive(Error, Debug)]
pub enum Error {
    #[error("node is not present in this `Context`")]
    NoSuchNode,
    #[error("`Context` is empty")]
    EmptyContext,
    #[error("`VecMap` is empty")]
    EmptyMap,
    #[error("unknown opcode {0}")]
    UnknownOpcode(String),
    #[error("empty file")]
    EmptyFile,
    #[error("The variable was not found in the tree")]
    NoSuchVar,
    #[error("The constant was not found in the tree")]
    NoSuchConst,
    #[error("Error occurred during i/o operation: {0}")]
    IoError(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Represents an index in the `Context::nodes` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Node(usize);
impl From<usize> for Node {
    fn from(v: usize) -> Self {
        Self(v)
    }
}
impl From<Node> for usize {
    fn from(v: Node) -> Self {
        v.0
    }
}

/// Represents an index in the `Context::vars` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VarNode(usize);
impl From<usize> for VarNode {
    fn from(v: usize) -> Self {
        Self(v)
    }
}
impl From<VarNode> for usize {
    fn from(v: VarNode) -> Self {
        v.0
    }
}

/// Represents an index in the `Context::consts` map
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConstNode(usize);
impl From<usize> for ConstNode {
    fn from(v: usize) -> Self {
        Self(v)
    }
}
impl From<ConstNode> for usize {
    fn from(v: ConstNode) -> Self {
        v.0
    }
}

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

use crate::vecmap::VecMap;

/// A `Context` holds a set of deduplicated constants, variables, and
/// operations.
///
/// It should be used like an arena allocator: it grows over time, then frees.
/// all of its contents when dropped.
#[derive(Debug, Default)]
pub struct Context {
    ops: VecMap<Op, Node>,
    vars: VecMap<String, VarNode>,
    consts: VecMap<OrderedFloat<f64>, ConstNode>,
}

impl Context {
    /// Build a new empty context
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of [Op] nodes in the context
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
        match self.vars.get_by_value(&s) {
            Some(v) => v,
            None => self.vars.insert(s),
        }
    }
    /// Looks up the [ConstNode] corresponding to the given value, inserting
    /// it if it doesn't already exist.
    fn get_const(&mut self, s: f64) -> ConstNode {
        let s = OrderedFloat(s);
        match self.consts.get_by_value(&s) {
            Some(v) => v,
            None => self.consts.insert(s),
        }
    }
    /// Looks up the [Node] corresponding to the given operation, inserting
    /// it if it doesn't already exist.
    fn get_node(&mut self, op: Op) -> Node {
        match self.ops.get_by_value(&op) {
            Some(v) => v,
            None => self.ops.insert(op),
        }
    }
    /// Erases the most recently added node from the tree. This should only
    /// be called to delete a temporary operation node, as it will invalidate
    /// any existing handles to the node; therefore, it's private.
    fn pop(&mut self) -> Result<(), Error> {
        self.ops.pop().map(|_| ())
    }

    /// Looks up the constant associated with the given node. If the node is
    /// invalid for this tree or is not `Op::Const`, returns `None`.
    fn const_value(&self, n: Node) -> Result<Option<f64>, Error> {
        match self.ops.get_by_index(n) {
            Some(Op::Const(c)) => match self.consts.get_by_index(*c) {
                Some(c) => Ok(Some(**c)),
                None => Err(Error::NoSuchConst),
            },
            Some(_) => Ok(None),
            _ => Err(Error::NoSuchNode),
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Primitives
    /// Constructs or finds a variable node named "X"
    /// ```
    /// use jitfive::{Context};
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// let v = ctx.eval_xyz(x, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn x(&mut self) -> Node {
        let v = self.get_var("X");
        self.get_node(Op::Var(v))
    }
    /// Constructs or finds a variable node named "Z"
    pub fn y(&mut self) -> Node {
        let v = self.get_var("Y");
        self.get_node(Op::Var(v))
    }
    /// Constructs or finds a variable node named "Z"
    pub fn z(&mut self) -> Node {
        let v = self.get_var("Z");
        self.get_node(Op::Var(v))
    }
    /// Returns a node representing the given constant value.
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let v = ctx.constant(3.0);
    /// assert_eq!(ctx.eval_xyz(v, 0.0, 0.0, 0.0).unwrap(), 3.0);
    /// ```
    pub fn constant(&mut self, f: f64) -> Node {
        let v = self.get_const(f);
        self.get_node(Op::Const(v))
    }

    ////////////////////////////////////////////////////////////////////////////
    // Helper functions to create nodes with constant folding
    /// Find or create a [Node] for the given unary operation, with constant
    /// folding.
    fn op_unary<F>(&mut self, a: Node, op: F) -> Result<Node, Error>
    where
        F: Fn(Node) -> Op,
    {
        let op_a = *self.ops.get_by_index(a).ok_or(Error::NoSuchNode)?;
        let n = self.get_node(op(a));
        let out = if matches!(op_a, Op::Const(_)) {
            let v = self.eval(n, &BTreeMap::new())?;
            self.pop().unwrap(); // removes `n`
            self.constant(v)
        } else {
            n
        };
        Ok(out)
    }
    /// Find or create a [Node] for the given binary operation, with constant
    /// folding.
    fn op_binary<F>(&mut self, a: Node, b: Node, op: F) -> Result<Node, Error>
    where
        F: Fn(Node, Node) -> Op,
    {
        let op_a = *self.ops.get_by_index(a).ok_or(Error::NoSuchNode)?;
        let op_b = *self.ops.get_by_index(b).ok_or(Error::NoSuchNode)?;
        let n = self.get_node(op(a, b));
        let out = if matches!((op_a, op_b), (Op::Const(_), Op::Const(_))) {
            let v = self.eval(n, &BTreeMap::new())?;
            self.pop().unwrap(); // removes `n`
            self.constant(v)
        } else {
            n
        };
        Ok(out)
    }
    /// Find or create a [Node] for the given commutative operation, with
    /// constant folding; deduplication is encouraged by sorting `a` and `b`.
    fn op_binary_commutative<F>(
        &mut self,
        a: Node,
        b: Node,
        op: F,
    ) -> Result<Node, Error>
    where
        F: Fn(Node, Node) -> Op,
    {
        self.op_binary(a.min(b), a.max(b), op)
    }

    /// Builds an addition node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let num = ctx.constant(1.0);
    /// let op = ctx.add(x, num).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn add(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        match (self.const_value(a)?, self.const_value(b)?) {
            (Some(zero), _) if zero == 0.0 => Ok(b),
            (_, Some(zero)) if zero == 0.0 => Ok(a),
            _ => self.op_binary_commutative(a, b, Op::Add),
        }
    }
    /// Builds an multiplication node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let num = ctx.constant(5.0);
    /// let op = ctx.mul(x, num).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 10.0);
    /// ```
    pub fn mul(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        match (self.const_value(a)?, self.const_value(b)?) {
            (Some(one), _) if one == 1.0 => Ok(b),
            (_, Some(one)) if one == 1.0 => Ok(a),
            _ => self.op_binary_commutative(a, b, Op::Mul),
        }
    }
    /// Builds an `min` node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let num = ctx.constant(5.0);
    /// let op = ctx.min(x, num).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn min(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        if a == b {
            Ok(a)
        } else {
            self.op_binary_commutative(a, b, Op::Min)
        }
    }
    /// Builds an `max` node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let num = ctx.constant(5.0);
    /// let op = ctx.max(x, num).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 5.0);
    /// ```
    pub fn max(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        if a == b {
            Ok(a)
        } else {
            self.op_binary_commutative(a, b, Op::Max)
        }
    }

    /// Builds a unary negation node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.neg(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, -2.0);
    /// ```
    pub fn neg(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Neg)
    }

    /// Builds a reciprocal node
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.recip(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 0.5);
    /// ```
    pub fn recip(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Recip)
    }

    /// Builds a node which calculates the absolute value of its input
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.abs(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// let v = ctx.eval_xyz(op, -2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn abs(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Abs)
    }

    /// Builds a node which calculates the square root of its input
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.sqrt(x).unwrap();
    /// let v = ctx.eval_xyz(op, 4.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn sqrt(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Sqrt)
    }

    /// Builds a node which calculates the sine of its input (in radians)
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.sin(x).unwrap();
    /// let v = ctx.eval_xyz(op, std::f64::consts::PI, 0.0, 0.0).unwrap();
    /// assert!(v.abs() < 1e-8); // approximately 0
    /// ```
    pub fn sin(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Sin)
    }
    /// Builds a node which calculates the cosine of its input (in radians)
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.cos(x).unwrap();
    /// let v = ctx.eval_xyz(op, std::f64::consts::PI * 2.0, 0.0, 0.0).unwrap();
    /// assert!((v - 1.0).abs() < 1e-8); // approximately 1.0
    /// ```
    pub fn cos(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Cos)
    }
    /// Builds a node which calculates the tangent of its input (in radians)
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.tan(x).unwrap();
    /// let v = ctx.eval_xyz(op, std::f64::consts::PI / 4.0, 0.0, 0.0).unwrap();
    /// assert!((v - 1.0).abs() < 1e-8); // approximately 1.0
    /// ```
    pub fn tan(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Tan)
    }
    /// Builds a node which calculates the inverse sine of its input, returning
    /// a value in radians.
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.asin(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert!((v - std::f64::consts::PI / 2.0).abs() < 1e-8);
    /// ```
    pub fn asin(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Asin)
    }
    /// Builds a node which calculates the inverse cosine of its input, returning
    /// a value in radians.
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.acos(x).unwrap();
    /// let v = ctx.eval_xyz(op, 0.0, 0.0, 0.0).unwrap();
    /// assert!((v - std::f64::consts::PI / 2.0).abs() < 1e-8);
    /// ```
    pub fn acos(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Acos)
    }
    /// Builds a node which calculates the inverse cosine of its input, returning
    /// a value in radians.
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.atan(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert!((v - std::f64::consts::PI / 4.0).abs() < 1e-8);
    /// ```
    pub fn atan(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Atan)
    }
    /// Builds a node which calculates the exponent (e^x) of the input
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.exp(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert!((v - std::f64::consts::E).abs() < 1e-8);
    /// ```
    pub fn exp(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Exp)
    }
    /// Builds a node which calculates the natural logaritm of its input
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.ln(x).unwrap();
    /// let v = ctx.eval_xyz(op, std::f64::consts::E, 0.0, 0.0).unwrap();
    /// assert!((v - 1.0).abs() < 1e-8);
    /// ```
    pub fn ln(&mut self, a: Node) -> Result<Node, Error> {
        self.op_unary(a, Op::Ln)
    }

    ////////////////////////////////////////////////////////////////////////////
    // Derived functions
    /// Builds a node which squares its input
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.square(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 4.0);
    /// ```
    pub fn square(&mut self, a: Node) -> Result<Node, Error> {
        self.mul(a, a)
    }
    /// Builds a node which performs subtraction. Under the hood, `a - b` is
    /// converted to `a + (-b)`
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.sub(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 3.0, 2.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn sub(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        let b = self.neg(b)?;
        self.add(a, b)
    }
    /// Builds a node which performs division. Under the hood, `a / b` is
    /// converted into `a * (1 / b)`.
    /// ```
    /// # let mut ctx = jitfive::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.div(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 3.0, 2.0, 0.0).unwrap();
    /// assert_eq!(v, 1.5);
    /// ```
    pub fn div(&mut self, a: Node, b: Node) -> Result<Node, Error> {
        let b = self.recip(b)?;
        self.mul(a, b)
    }

    ////////////////////////////////////////////////////////////////////////////
    pub fn eval_xyz(
        &self,
        root: Node,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<f64, Error> {
        let vars = [("X", x), ("Y", y), ("Z", z)]
            .into_iter()
            .map(|(a, b)| (a.to_string(), b))
            .collect();
        self.eval(root, &vars)
    }
    pub fn eval(
        &self,
        root: Node,
        vars: &BTreeMap<String, f64>,
    ) -> Result<f64, Error> {
        let mut cache = vec![None; self.ops.len()];
        self.eval_inner(root, vars, &mut cache)
    }
    fn eval_inner(
        &self,
        node: Node,
        vars: &BTreeMap<String, f64>,
        cache: &mut [Option<f64>],
    ) -> Result<f64, Error> {
        if let Some(v) = cache[node.0] {
            return Ok(v);
        }
        let mut get = |n: Node| self.eval_inner(n, vars, cache);
        let v = match self.ops.get_by_index(node).ok_or(Error::NoSuchNode)? {
            Op::Var(v) => {
                let var_name = self.vars.get_by_index(*v).unwrap();
                *vars.get(var_name).unwrap()
            }
            Op::Const(c) => self.consts.get_by_index(*c).unwrap().0,

            Op::Add(a, b) => get(*a)? + get(*b)?,
            Op::Mul(a, b) => get(*a)? * get(*b)?,
            Op::Min(a, b) => get(*a)?.min(get(*b)?),
            Op::Max(a, b) => get(*a)?.max(get(*b)?),

            // Unary operations
            Op::Neg(a) => -get(*a)?,
            Op::Abs(a) => get(*a)?.abs(),
            Op::Recip(a) => 1.0 / get(*a)?,

            // Transcendental functions
            Op::Sqrt(a) => get(*a)?.sqrt(),
            Op::Sin(a) => get(*a)?.sin(),
            Op::Cos(a) => get(*a)?.cos(),
            Op::Tan(a) => get(*a)?.tan(),
            Op::Asin(a) => get(*a)?.asin(),
            Op::Acos(a) => get(*a)?.acos(),
            Op::Atan(a) => get(*a)?.atan(),
            Op::Exp(a) => get(*a)?.exp(),
            Op::Ln(a) => get(*a)?.ln(),
        };

        cache[node.0] = Some(v);
        Ok(v)
    }

    /// Parses a flat text representation of a math tree. For example, the
    /// circle `(- (+ (square x) (square y)) 1)` can be parsed from
    /// ```
    /// # use jitfive::Context;
    /// let txt = "
    /// 0x600000b90000 var-x
    /// 0x600000b900a0 square 0x600000b90000
    /// 0x600000b90050 var-y
    /// 0x600000b900f0 square 0x600000b90050
    /// 0x600000b90140 add 0x600000b900a0 0x600000b900f0
    /// 0x600000b90190 sqrt 0x600000b90140
    /// 0x600000b901e0 const 1
    /// ";
    /// let (ctx, _node) = Context::from_text(&mut txt.as_bytes()).unwrap();
    /// assert_eq!(ctx.len(), 7);
    /// ```
    ///
    /// This representation is loosely defined and only intended for use in
    /// quick experiments.
    pub fn from_text<R: Read>(r: &mut R) -> Result<(Self, Node), Error> {
        use parse_int::parse;

        let reader = BufReader::new(r);
        let mut ctx = Self::new();
        let mut seen = BTreeMap::new();
        let mut last = None;

        for line in reader.lines().map(|line| line.unwrap()) {
            if line.is_empty() {
                continue;
            }
            let mut iter = line.split_whitespace();
            let i: u64 = parse(iter.next().unwrap()).unwrap();
            let opcode = iter.next().unwrap();

            let mut pop = || seen[&parse(iter.next().unwrap()).unwrap()];
            let node = match opcode {
                "const" => ctx.constant(iter.next().unwrap().parse().unwrap()),
                "var-x" => ctx.x(),
                "var-y" => ctx.y(),
                "var-z" => ctx.z(),
                "abs" => ctx.abs(pop())?,
                "neg" => ctx.neg(pop())?,
                "sqrt" => ctx.sqrt(pop())?,
                "square" => ctx.square(pop())?,
                "add" => ctx.add(pop(), pop())?,
                "mul" => ctx.mul(pop(), pop())?,
                "min" => ctx.min(pop(), pop())?,
                "max" => ctx.max(pop(), pop())?,
                "div" => ctx.div(pop(), pop())?,
                "sub" => ctx.sub(pop(), pop())?,
                op => return Err(Error::UnknownOpcode(op.to_owned())),
            };
            seen.insert(i, node);
            last = Some(node);
        }
        match last {
            Some(node) => Ok((ctx, node)),
            None => Err(Error::EmptyFile),
        }
    }

    /// Converts the given tree into a GraphViz drawing
    pub fn to_dot<W: Write>(
        &self,
        w: &mut W,
        _root: Node,
    ) -> Result<(), Error> {
        writeln!(w, "digraph mygraph {{")?;
        for (op, node) in self.ops.iter() {
            // Write node label
            write!(w, r#"n{} [label = ""#, node.0)?;
            match op {
                Op::Const(c) => {
                    let v = self
                        .consts
                        .get_by_index(*c)
                        .ok_or(Error::NoSuchConst)?;
                    write!(w, "{}", v)
                }
                Op::Var(v) => {
                    let v =
                        self.vars.get_by_index(*v).ok_or(Error::NoSuchVar)?;
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

    /// Does as-simple-as-possible 2D rendering, drawing the square region
    /// between [-1, +1] on both axes. Returns a `Vec<bool>` of size
    /// `scale * scale`, with pixels set based on sign.
    pub fn render_2d(
        &self,
        root: Node,
        scale: usize,
    ) -> Result<Vec<bool>, Error> {
        let mut out = Vec::with_capacity(scale * scale);
        let div = (scale - 1) as f64;
        for i in 0..scale {
            let y = -(-1.0 + 2.0 * (i as f64) / div);
            for j in 0..scale {
                let x = -1.0 + 2.0 * (j as f64) / div;
                let v = self.eval_xyz(root, x, y, 0.0)?;
                out.push(v <= 0.0);
            }
        }
        Ok(out)
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
        assert_eq!(ctx.const_value(a).unwrap(), Some(1.0));
        assert_eq!(ctx.const_value(x1).unwrap(), None);

        let c = ctx.add(a, b).unwrap();
        assert_eq!(ctx.const_value(c).unwrap(), Some(2.0));

        let c = ctx.neg(c).unwrap();
        assert_eq!(ctx.const_value(c).unwrap(), Some(-2.0));
    }

    #[test]
    fn test_constant_folding() {
        let mut ctx = Context::new();
        let a = ctx.constant(1.0);
        assert_eq!(ctx.len(), 1);
        println!("{:?}", ctx);
        let b = ctx.constant(-1.0);
        println!("{:?}", ctx);
        assert_eq!(ctx.len(), 2);
        let _ = ctx.add(a, b);
        assert_eq!(ctx.len(), 3);
        let _ = ctx.add(a, b);
        assert_eq!(ctx.len(), 3);
        let _ = ctx.mul(a, b);
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn test_eval() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let v = ctx.add(x, y).unwrap();

        assert_eq!(
            ctx.eval(
                v,
                &[("X".to_string(), 1.0), ("Y".to_string(), 2.0)]
                    .into_iter()
                    .collect()
            )
            .unwrap(),
            3.0
        );
        assert_eq!(ctx.eval_xyz(v, 2.0, 3.0, 0.0).unwrap(), 5.0);
    }
}
