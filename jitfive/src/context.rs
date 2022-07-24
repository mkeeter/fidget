use crate::{
    error::Error,
    indexed::{IndexMap, IndexVec},
    op::{Node, Op, VarNode},
};

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};

use ordered_float::OrderedFloat;

/// A `Context` holds a set of deduplicated constants, variables, and
/// operations.
///
/// It should be used like an arena allocator: it grows over time, then frees
/// all of its contents when dropped.
#[derive(Debug, Default)]
pub struct Context {
    ops: IndexMap<Op, Node>,
    vars: IndexMap<String, VarNode>,
}

impl Context {
    /// Build a new empty context
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of `Op` nodes in the context
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Erases the most recently added node from the tree.
    ///
    /// A few caveats apply, so this must be used with caution:
    /// - Existing handles to the node will be invalidated
    /// - The most recently added node must be unique
    ///
    /// In practice, this is only used to delete temporary operation nodes
    /// during constant folding.  Such nodes which have no handles (because
    /// they are never returned) and are guaranteed to be unique (because we
    /// never store them persistently).
    fn pop(&mut self) -> Result<(), Error> {
        self.ops.pop().map(|_| ())
    }

    /// Looks up the constant associated with the given node. If the node is
    /// invalid for this tree, returns an error; if the node is not a constant,
    /// returns `Ok(None)`.
    pub fn const_value(&self, n: Node) -> Result<Option<f64>, Error> {
        match self.ops.get_by_index(n) {
            Some(Op::Const(c)) => Ok(Some(c.0)),
            Some(_) => Ok(None),
            _ => Err(Error::BadNode),
        }
    }

    /// Looks up the variable name associated with the given node. If the node
    /// is invalid for this tree, returns an error; if the node is not a
    /// `Op::Var`, returns `Ok(None)`.
    pub fn var_name(&self, n: Node) -> Result<Option<&str>, Error> {
        match self.ops.get_by_index(n) {
            Some(Op::Var(c)) => self.get_var_by_index(*c).map(Some),
            Some(_) => Ok(None),
            _ => Err(Error::BadNode),
        }
    }

    /// Looks up the variable name associated with the given `VarNode`
    pub fn get_var_by_index(&self, n: VarNode) -> Result<&str, Error> {
        match self.vars.get_by_index(n) {
            Some(c) => Ok(c),
            None => Err(Error::BadVar),
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Primitives
    /// Constructs or finds a variable node named "X"
    /// ```
    /// # use jitfive::context::Context;
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// let v = ctx.eval_xyz(x, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn x(&mut self) -> Node {
        let v = self.vars.insert(String::from("X"));
        self.ops.insert(Op::Var(v))
    }

    /// Constructs or finds a variable node named "Y"
    pub fn y(&mut self) -> Node {
        let v = self.vars.insert(String::from("Y"));
        self.ops.insert(Op::Var(v))
    }

    /// Constructs or finds a variable node named "Z"
    pub fn z(&mut self) -> Node {
        let v = self.vars.insert(String::from("Z"));
        self.ops.insert(Op::Var(v))
    }

    /// Returns a node representing the given constant value.
    /// ```
    /// # let mut ctx = jitfive::context::Context::new();
    /// let v = ctx.constant(3.0);
    /// assert_eq!(ctx.eval_xyz(v, 0.0, 0.0, 0.0).unwrap(), 3.0);
    /// ```
    pub fn constant(&mut self, f: f64) -> Node {
        self.ops.insert(Op::Const(OrderedFloat(f)))
    }

    ////////////////////////////////////////////////////////////////////////////
    // Helper functions to create nodes with constant folding
    /// Find or create a [Node] for the given unary operation, with constant
    /// folding.
    fn op_unary<F>(&mut self, a: Node, op: F) -> Result<Node, Error>
    where
        F: Fn(Node) -> Op,
    {
        let op_a = *self.ops.get_by_index(a).ok_or(Error::BadNode)?;
        let n = self.ops.insert(op(a));
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
        let op_a = *self.ops.get_by_index(a).ok_or(Error::BadNode)?;
        let op_b = *self.ops.get_by_index(b).ok_or(Error::BadNode)?;

        // This call to `insert` should always insert the node, because we
        // don't permanently store operations in the tree that could be
        // constant-folded (indeed, we pop the node right afterwards)
        let n = self.ops.insert(op(a, b));
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
            self.op_binary_commutative(a, b, |lhs, rhs| Op::Min(lhs, rhs, ()))
        }
    }
    /// Builds an `max` node
    /// ```
    /// # let mut ctx = jitfive::context::Context::new();
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
            self.op_binary_commutative(a, b, |lhs, rhs| Op::Max(lhs, rhs, ()))
        }
    }

    /// Builds a unary negation node
    /// ```
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// # let mut ctx = jitfive::context::Context::new();
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
    /// Evaluates the given node with the provided values for X, Y, and Z.
    ///
    /// ```
    /// # let mut ctx = jitfive::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let z = ctx.z();
    /// let op = ctx.mul(x, y).unwrap();
    /// let op = ctx.div(op, z).unwrap();
    /// let v = ctx.eval_xyz(op, 3.0, 5.0, 2.0).unwrap();
    /// assert_eq!(v, 7.5); // (3.0 * 5.0) / 2.0
    /// ```
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

    /// Evaluates the given node with a generic set of variables
    pub fn eval(
        &self,
        root: Node,
        vars: &BTreeMap<String, f64>,
    ) -> Result<f64, Error> {
        let mut cache = vec![None; self.ops.len()].into();
        self.eval_inner(root, vars, &mut cache)
    }

    fn eval_inner(
        &self,
        node: Node,
        vars: &BTreeMap<String, f64>,
        cache: &mut IndexVec<Option<f64>, Node>,
    ) -> Result<f64, Error> {
        if let Some(v) = cache[node] {
            return Ok(v);
        }
        let mut get = |n: Node| self.eval_inner(n, vars, cache);
        let v = match self.ops.get_by_index(node).ok_or(Error::BadNode)? {
            Op::Var(v) => {
                let var_name = self.vars.get_by_index(*v).unwrap();
                *vars.get(var_name).unwrap()
            }
            Op::Const(c) => c.0,

            Op::Add(a, b) => get(*a)? + get(*b)?,
            Op::Mul(a, b) => get(*a)? * get(*b)?,
            Op::Min(a, b, _) => get(*a)?.min(get(*b)?),
            Op::Max(a, b, _) => get(*a)?.max(get(*b)?),

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

        cache[node] = Some(v);
        Ok(v)
    }

    /// Parses a flat text representation of a math tree. For example, the
    /// circle `(- (+ (square x) (square y)) 1)` can be parsed from
    /// ```
    /// # use jitfive::context::Context;
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
        let reader = BufReader::new(r);
        let mut ctx = Self::new();
        let mut seen = BTreeMap::new();
        let mut last = None;

        for line in reader.lines().map(|line| line.unwrap()) {
            if line.is_empty() {
                continue;
            }
            let mut iter = line.split_whitespace();
            let i: String = iter.next().unwrap().to_owned();
            let opcode = iter.next().unwrap();

            let mut pop = || {
                let txt = iter.next().unwrap();
                seen.get(txt)
                    .cloned()
                    .ok_or_else(|| Error::UnknownVariable(txt.to_string()))
            };
            let node = match opcode {
                "const" => ctx.constant(iter.next().unwrap().parse().unwrap()),
                "var-x" => ctx.x(),
                "var-y" => ctx.y(),
                "var-z" => ctx.z(),
                "abs" => ctx.abs(pop()?)?,
                "neg" => ctx.neg(pop()?)?,
                "sqrt" => ctx.sqrt(pop()?)?,
                "square" => ctx.square(pop()?)?,
                "add" => ctx.add(pop()?, pop()?)?,
                "mul" => ctx.mul(pop()?, pop()?)?,
                "min" => ctx.min(pop()?, pop()?)?,
                "max" => ctx.max(pop()?, pop()?)?,
                "div" => ctx.div(pop()?, pop()?)?,
                "sub" => ctx.sub(pop()?, pop()?)?,
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
        _root: Node,
        w: &mut W,
    ) -> Result<(), Error> {
        writeln!(w, "digraph mygraph {{")?;
        for node in self.ops.keys() {
            self.write_node_dot(w, node)?;
            self.write_edges_dot(w, node)?;
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
        scale: u32,
    ) -> Result<Vec<bool>, Error> {
        let mut out = Vec::with_capacity((scale * scale) as usize);
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
    pub(crate) fn write_node_dot<W: Write>(
        &self,
        w: &mut W,
        node: Node,
    ) -> Result<(), Error> {
        let op = self.ops.get_by_index(node).unwrap();
        // Write node label
        write!(w, r#"n{} [label = ""#, node.dot_name())?;
        match op {
            Op::Const(c) => {
                write!(w, "{}", c)
            }
            Op::Var(v) => {
                let v = self.vars.get_by_index(*v).ok_or(Error::BadVar)?;
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
        writeln!(
            w,
            r#"" color="{0}1" shape="{1}" fontcolor="{0}4"]"#,
            op.dot_node_color(),
            op.dot_node_shape()
        )?;
        Ok(())
    }

    pub(crate) fn write_edges_dot<W: Write>(
        &self,
        w: &mut W,
        node: Node,
    ) -> Result<(), Error> {
        let op = self.ops.get_by_index(node).unwrap();
        let edge_color = format!("{}4", op.dot_node_color());
        match op {
            Op::Add(a, b)
            | Op::Mul(a, b)
            | Op::Min(a, b, _)
            | Op::Max(a, b, _) => {
                writeln!(
                    w,
                    "n{0} -> n{1} [color=\"{3}\"];\
                         n{0} -> n{2} [color=\"{3}\"]",
                    node.dot_name(),
                    a.dot_name(),
                    b.dot_name(),
                    edge_color,
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
                node.dot_name(),
                a.dot_name(),
                edge_color
            ),

            Op::Var(..) | Op::Const(..) => Ok(()),
        }?;
        Ok(())
    }

    /// Looks up an operation by `Node` handle
    pub(crate) fn get_op(&self, node: Node) -> Option<&Op> {
        self.ops.get_by_index(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // This can't be in a doctest, because it uses a pub(crate) function
    #[test]
    fn test_get_op() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let op_x = ctx.get_op(x).unwrap();
        assert!(matches!(op_x, Op::Var(_)));
    }
}
