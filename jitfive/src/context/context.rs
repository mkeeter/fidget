use crate::{
    context::{
        indexed::{define_index, Index, IndexMap, IndexVec},
        BinaryOpcode, Op, UnaryOpcode,
    },
    error::Error,
    tape::{SsaTapeBuilder, Tape},
};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::io::{BufRead, BufReader, Read};

use ordered_float::OrderedFloat;

define_index!(Node, "An index in the `Context::ops` map");
define_index!(VarNode, "An index in the `Context::vars` map");

impl Node {
    pub fn dot_name(&self) -> String {
        format!("n{}", self.0)
    }
}

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
    /// Returns the number of [`Op`](crate::context::Op) nodes in the context
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Checks whether the context is empty
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
    fn op_unary(&mut self, a: Node, op: UnaryOpcode) -> Result<Node, Error> {
        let op_a = *self.ops.get_by_index(a).ok_or(Error::BadNode)?;
        let n = self.ops.insert(Op::Unary(op, a));
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
    fn op_binary(
        &mut self,
        a: Node,
        b: Node,
        op: BinaryOpcode,
    ) -> Result<Node, Error> {
        self.op_binary_f(a, b, |lhs, rhs| Op::Binary(op, lhs, rhs))
    }

    /// Find or create a [Node] for a generic binary operation (represented by a
    /// thunk), with constant folding.
    fn op_binary_f<F>(&mut self, a: Node, b: Node, f: F) -> Result<Node, Error>
    where
        F: Fn(Node, Node) -> Op,
    {
        let op_a = *self.ops.get_by_index(a).ok_or(Error::BadNode)?;
        let op_b = *self.ops.get_by_index(b).ok_or(Error::BadNode)?;

        // This call to `insert` should always insert the node, because we
        // don't permanently store operations in the tree that could be
        // constant-folded (indeed, we pop the node right afterwards)
        let n = self.ops.insert(f(a, b));
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
    fn op_binary_commutative(
        &mut self,
        a: Node,
        b: Node,
        op: BinaryOpcode,
    ) -> Result<Node, Error> {
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
        if a == b {
            let two = self.constant(2.0);
            self.mul(a, two)
        } else {
            match (self.const_value(a)?, self.const_value(b)?) {
                (Some(zero), _) if zero == 0.0 => Ok(b),
                (_, Some(zero)) if zero == 0.0 => Ok(a),
                _ => self.op_binary_commutative(a, b, BinaryOpcode::Add),
            }
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
        if a == b {
            self.square(a)
        } else {
            match (self.const_value(a)?, self.const_value(b)?) {
                (Some(one), _) if one == 1.0 => Ok(b),
                (_, Some(one)) if one == 1.0 => Ok(a),
                _ => self.op_binary_commutative(a, b, BinaryOpcode::Mul),
            }
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
            self.op_binary_commutative(a, b, BinaryOpcode::Min)
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
            self.op_binary_commutative(a, b, BinaryOpcode::Max)
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
        self.op_unary(a, UnaryOpcode::Neg)
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
        self.op_unary(a, UnaryOpcode::Recip)
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
        self.op_unary(a, UnaryOpcode::Abs)
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
        self.op_unary(a, UnaryOpcode::Sqrt)
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
        self.op_unary(a, UnaryOpcode::Square)
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
        self.op_binary(a, b, BinaryOpcode::Sub)
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
    pub fn get_tape(&self, root: Node, reg_limit: u8) -> Tape {
        let mut parent_count: BTreeMap<Node, usize> = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut todo = vec![root];
        let mut builder = SsaTapeBuilder::new();

        // Accumulate parent counts and declare all the nodes into the builder
        while let Some(node) = todo.pop() {
            if !seen.insert(node) {
                continue;
            }
            let op = self.get_op(node).unwrap();
            builder.declare_node(node, *op);
            for child in op.iter_children() {
                *parent_count.entry(child).or_default() += 1;
                todo.push(child);
            }
        }

        // Now that we've populated our parents, flatten the graph
        let mut todo = vec![root];
        let mut seen = BTreeSet::new();
        while let Some(node) = todo.pop() {
            if *parent_count.get(&node).unwrap_or(&0) > 0 || !seen.insert(node)
            {
                continue;
            }
            let op = self.get_op(node).unwrap();
            for child in op.iter_children() {
                todo.push(child);
                *parent_count.get_mut(&child).unwrap() -= 1;
            }
            builder.step(node, *op, self);
        }
        let ssa_tape = builder.finish();
        Tape::from_ssa(ssa_tape, reg_limit)
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

            Op::Binary(op, a, b) => {
                let a = get(*a)?;
                let b = get(*b)?;
                match op {
                    BinaryOpcode::Add => a + b,
                    BinaryOpcode::Mul => a * b,
                    BinaryOpcode::Sub => a - b,
                    BinaryOpcode::Min => a.min(b),
                    BinaryOpcode::Max => a.max(b),
                }
            }

            // Unary operations
            Op::Unary(op, a) => {
                let a = get(*a)?;
                match op {
                    UnaryOpcode::Neg => -a,
                    UnaryOpcode::Abs => a.abs(),
                    UnaryOpcode::Recip => 1.0 / a,
                    UnaryOpcode::Sqrt => a.sqrt(),
                    UnaryOpcode::Square => a * a,
                }
            }
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

    /// Converts the entire context into a GraphViz drawing
    pub fn dot(&self) -> String {
        let mut out = "digraph mygraph{\n".to_owned();
        for node in self.ops.keys() {
            let op = self.ops.get_by_index(node).unwrap();
            out += &self.dot_node(node);
            out += &op.dot_edges(node);
        }
        out += "}\n";
        out
    }

    /// Converts the given node into a GraphViz node
    ///
    /// (this is a local function instead of a function on `Op` because it
    ///  requires looking up variables by name)
    fn dot_node(&self, i: Node) -> String {
        let mut out = format!(r#"n{} [label = ""#, i.get());
        let op = self.ops.get_by_index(i).unwrap();
        match op {
            Op::Const(c) => write!(out, "{}", c).unwrap(),
            Op::Var(v) => {
                let v = self.vars.get_by_index(*v).unwrap();
                out += v;
            }
            Op::Binary(op, ..) => match op {
                BinaryOpcode::Add => out += "add",
                BinaryOpcode::Mul => out += "mul",
                BinaryOpcode::Sub => out += "sub",
                BinaryOpcode::Min => out += "min",
                BinaryOpcode::Max => out += "max",
            },
            Op::Unary(op, ..) => match op {
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
            op.dot_node_color(),
            op.dot_node_shape()
        )
        .unwrap();
        out
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
