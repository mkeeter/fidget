//! Infrastructure for representing math expressions as graphs
mod indexed;
mod op;

#[cfg(test)]
pub(crate) mod bound;

use indexed::{define_index, Index, IndexMap, IndexVec};
pub use op::{BinaryOpcode, Op, UnaryOpcode};

use crate::{eval::Family, vm::Tape, Error};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::io::{BufRead, BufReader, Read};

use ordered_float::OrderedFloat;

define_index!(Node, "An index in the `Context::ops` map");
define_index!(VarNode, "An index in the `Context::vars` map");

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

    /// Clears the context
    ///
    /// All [`Node`](Node) and [`VarNode`](VarNode) handles from this context
    /// are invalidated.
    ///
    /// ```
    /// # use fidget::context::Context;
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// ctx.clear();
    /// assert!(ctx.eval_xyz(x, 1.0, 0.0, 0.0).is_err());
    /// ```
    pub fn clear(&mut self) {
        self.ops.clear();
        self.vars.clear();
    }

    /// Returns the number of [`Op`] nodes in the context
    ///
    /// ```
    /// # use fidget::context::Context;
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// assert_eq!(ctx.len(), 1);
    /// let y = ctx.y();
    /// assert_eq!(ctx.len(), 2);
    /// ctx.clear();
    /// assert_eq!(ctx.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Checks whether the context is empty
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Checks whether the given [`Node`](Node) is valid in this context
    fn check_node(&self, node: Node) -> Result<(), Error> {
        self.get_op(node).ok_or(Error::BadNode).map(|_| ())
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

    /// Looks up the constant associated with the given node.
    ///
    /// If the node is invalid for this tree, returns an error; if the node is
    /// not a constant, returns `Ok(None)`.
    pub fn const_value(&self, n: Node) -> Result<Option<f64>, Error> {
        match self.get_op(n) {
            Some(Op::Const(c)) => Ok(Some(c.0)),
            Some(_) => Ok(None),
            _ => Err(Error::BadNode),
        }
    }

    /// Looks up the variable name associated with the given node.
    ///
    /// If the node is invalid for this tree, returns an error; if the node is
    /// not a `Op::Var` or `Op::Input`, returns `Ok(None)`.
    pub fn var_name(&self, n: Node) -> Result<Option<&str>, Error> {
        match self.get_op(n) {
            Some(Op::Var(c) | Op::Input(c)) => {
                self.get_var_by_index(*c).map(Some)
            }
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
    /// # use fidget::context::Context;
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// let v = ctx.eval_xyz(x, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn x(&mut self) -> Node {
        let v = self.vars.insert(String::from("X"));
        self.ops.insert(Op::Input(v))
    }

    /// Constructs or finds a variable node named "Y"
    pub fn y(&mut self) -> Node {
        let v = self.vars.insert(String::from("Y"));
        self.ops.insert(Op::Input(v))
    }

    /// Constructs or finds a variable node named "Z"
    pub fn z(&mut self) -> Node {
        let v = self.vars.insert(String::from("Z"));
        self.ops.insert(Op::Input(v))
    }

    /// Returns a variable with the provided name.
    ///
    /// If a variable already exists with this name, then it is returned.
    pub fn var(&mut self, name: &str) -> Result<Node, Error> {
        let name = self.check_var_name(name)?;
        let v = self.vars.insert(name);
        Ok(self.ops.insert(Op::Var(v)))
    }

    fn check_var_name(&self, name: &str) -> Result<String, Error> {
        if matches!(name, "X" | "Y" | "Z") {
            return Err(Error::ReservedName);
        }
        Ok(String::from(name))
    }

    /// Returns a node representing the given constant value.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
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
        let op_a = *self.get_op(a).ok_or(Error::BadNode)?;
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
        let op_a = *self.get_op(a).ok_or(Error::BadNode)?;
        let op_b = *self.get_op(b).ok_or(Error::BadNode)?;

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
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.add(x, 1.0).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn add<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a: Node = a.into_node(self)?;
        let b: Node = b.into_node(self)?;
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
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.mul(x, 5.0).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 10.0);
    /// ```
    pub fn mul<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;
        if a == b {
            self.square(a)
        } else {
            match (self.const_value(a)?, self.const_value(b)?) {
                (Some(one), _) if one == 1.0 => Ok(b),
                (_, Some(one)) if one == 1.0 => Ok(a),
                (Some(zero), _) if zero == 0.0 => Ok(a),
                (_, Some(zero)) if zero == 0.0 => Ok(b),
                _ => self.op_binary_commutative(a, b, BinaryOpcode::Mul),
            }
        }
    }

    /// Builds an `min` node
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.min(x, 5.0).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn min<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;
        if a == b {
            Ok(a)
        } else {
            self.op_binary_commutative(a, b, BinaryOpcode::Min)
        }
    }
    /// Builds an `max` node
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.max(x, 5.0).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 5.0);
    /// ```
    pub fn max<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;
        if a == b {
            Ok(a)
        } else {
            self.op_binary_commutative(a, b, BinaryOpcode::Max)
        }
    }

    /// Builds a unary negation node
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.neg(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, -2.0);
    /// ```
    pub fn neg<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Neg)
    }

    /// Builds a reciprocal node
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.recip(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 0.5);
    /// ```
    pub fn recip<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Recip)
    }

    /// Builds a node which calculates the absolute value of its input
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.abs(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// let v = ctx.eval_xyz(op, -2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn abs<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Abs)
    }

    /// Builds a node which calculates the square root of its input
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.sqrt(x).unwrap();
    /// let v = ctx.eval_xyz(op, 4.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn sqrt<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Sqrt)
    }

    ////////////////////////////////////////////////////////////////////////////
    // Derived functions
    /// Builds a node which squares its input
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.square(x).unwrap();
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 4.0);
    /// ```
    pub fn square<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Square)
    }

    /// Builds a node which performs subtraction.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.sub(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 3.0, 2.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn sub<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;

        match (self.const_value(a)?, self.const_value(b)?) {
            (Some(zero), _) if zero == 0.0 => self.neg(b),
            (_, Some(zero)) if zero == 0.0 => Ok(a),
            _ => self.op_binary(a, b, BinaryOpcode::Sub),
        }
    }

    /// Builds a node which performs division. Under the hood, `a / b` is
    /// converted into `a * (1 / b)`.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.div(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 3.0, 2.0, 0.0).unwrap();
    /// assert_eq!(v, 1.5);
    /// ```
    pub fn div<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;

        match (self.const_value(a)?, self.const_value(b)?) {
            (Some(zero), _) if zero == 0.0 => Ok(a),
            (_, Some(one)) if one == 1.0 => Ok(a),
            _ => self.op_binary(a, b, BinaryOpcode::Div),
        }
    }

    /// Flattens a subtree of the graph into straight-line code.
    ///
    /// The resulting tape uses `E::REG_LIMIT` registers; if more memory is
    /// required, it includes
    /// [`vm::Op::Load` / `vm::Op::Store`](crate::vm::Op) operations.
    ///
    /// This should always succeed unless the `root` is from a different
    /// `Context`, in which case `Error::BadNode` will be returned.
    pub fn get_tape<E: Family>(&self, root: Node) -> Result<Tape<E>, Error> {
        // TODO: make inlining a parameter here?
        // TODO buildy should return an error
        let data = crate::vm::build::buildy(self, root, 0)?;
        Ok(data.into())
    }

    ////////////////////////////////////////////////////////////////////////////

    /// Remaps the X, Y, Z nodes to the given values
    pub fn remap_xyz(
        &mut self,
        root: Node,
        xyz: [Node; 3],
    ) -> Result<Node, Error> {
        self.check_node(root)?;
        xyz.iter().try_for_each(|x| self.check_node(*x))?;

        let mut done = [self.x(), self.y(), self.z()]
            .into_iter()
            .zip(xyz)
            .collect::<BTreeMap<_, _>>();

        // Depth-first recursion on the heap, to protect against stack overflows
        enum Action {
            Down,
            Up,
        }

        let mut todo = vec![(Action::Down, root)];
        let mut seen = BTreeSet::new();
        while let Some((action, node)) = todo.pop() {
            match action {
                Action::Down => {
                    if !seen.insert(node) {
                        continue;
                    }
                    todo.push((Action::Up, node));
                    todo.extend(
                        self.get_op(node)
                            .unwrap()
                            .iter_children()
                            .map(|c| (Action::Down, c)),
                    );
                }
                Action::Up => {
                    let r = match self.get_op(node).unwrap() {
                        Op::Binary(op, lhs, rhs) => {
                            let a = done.get(lhs).unwrap();
                            let b = done.get(rhs).unwrap();
                            self.op_binary(*a, *b, *op).unwrap()
                        }
                        Op::Unary(op, arg) => {
                            let a = done.get(arg).unwrap();
                            self.op_unary(*a, *op).unwrap()
                        }
                        Op::Var(..) | Op::Const(..) => node,
                        Op::Input(..) => *done.get(&node).unwrap_or(&node),
                    };
                    done.insert(node, r);
                }
            }
        }
        Ok(*done.get(&root).unwrap())
    }

    ////////////////////////////////////////////////////////////////////////////
    /// Evaluates the given node with the provided values for X, Y, and Z.
    ///
    /// This is extremely inefficient; consider calling
    /// [`get_tape`](Self::get_tape) and building an evaluator instead.
    ///
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
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
    ///
    /// This is extremely inefficient; consider calling
    /// [`get_tape`](Self::get_tape) and building an evaluator instead.
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
        if node.0 >= cache.len() {
            return Err(Error::BadNode);
        }
        if let Some(v) = cache[node] {
            return Ok(v);
        }
        let mut get = |n: Node| self.eval_inner(n, vars, cache);
        let v = match self.get_op(node).ok_or(Error::BadNode)? {
            Op::Var(v) | Op::Input(v) => {
                let var_name = self.vars.get_by_index(*v).unwrap();
                *vars.get(var_name).unwrap()
            }
            Op::Const(c) => c.0,

            Op::Binary(op, a, b) => {
                let a = get(*a)?;
                let b = get(*b)?;
                match op {
                    BinaryOpcode::Add => a + b,
                    BinaryOpcode::Sub => a - b,
                    BinaryOpcode::Mul => a * b,
                    BinaryOpcode::Div => a / b,
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
    /// # use fidget::context::Context;
    /// let txt = "
    /// ## This is a comment!
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
    pub fn from_text<R: Read>(r: R) -> Result<(Self, Node), Error> {
        let reader = BufReader::new(r);
        let mut ctx = Self::new();
        let mut seen = BTreeMap::new();
        let mut last = None;

        for line in reader.lines().map(|line| line.unwrap()) {
            if line.is_empty() || line.starts_with('#') {
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
            let op = self.get_op(node).unwrap();
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
    pub fn dot_node(&self, i: Node) -> String {
        let mut out = format!(r#"n{} [label = ""#, i.get());
        let op = self.get_op(i).unwrap();
        match op {
            Op::Const(c) => write!(out, "{}", c).unwrap(),
            Op::Var(v) | Op::Input(v) => {
                let v = self.vars.get_by_index(*v).unwrap();
                out += v;
            }
            Op::Binary(op, ..) => match op {
                BinaryOpcode::Add => out += "add",
                BinaryOpcode::Sub => out += "sub",
                BinaryOpcode::Mul => out += "mul",
                BinaryOpcode::Div => out += "div",
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
    pub fn get_op(&self, node: Node) -> Option<&Op> {
        self.ops.get_by_index(node)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Helper trait for things that can be converted into a
/// [`Node`](Node) given a [`Context`](Context).
///
/// This trait allows you to write
/// ```
/// # let mut ctx = fidget::context::Context::new();
/// let x = ctx.x();
/// let sum = ctx.add(x, 1.0).unwrap();
/// ```
/// instead of the more verbose
/// ```
/// # let mut ctx = fidget::context::Context::new();
/// let x = ctx.x();
/// let num = ctx.constant(1.0);
/// let sum = ctx.add(x, num).unwrap();
/// ```
pub trait IntoNode {
    /// Converts the given values into a node
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error>;
}

impl IntoNode for Node {
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error> {
        ctx.check_node(self)?;
        Ok(self)
    }
}

impl IntoNode for f32 {
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error> {
        Ok(ctx.constant(self as f64))
    }
}

impl IntoNode for f64 {
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error> {
        Ok(ctx.constant(self))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    // This can't be in a doctest, because it uses a private function
    #[test]
    fn test_get_op() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let op_x = ctx.get_op(x).unwrap();
        assert!(matches!(op_x, Op::Input(_)));
    }

    #[test]
    fn test_ring() {
        let mut ctx = Context::new();
        let c0 = ctx.constant(0.5);
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r = ctx.add(x2, y2).unwrap();
        let c6 = ctx.sub(r, c0).unwrap();
        let c7 = ctx.constant(0.25);
        let c8 = ctx.sub(c7, r).unwrap();
        let c9 = ctx.max(c8, c6).unwrap();

        let tape = ctx.get_tape::<crate::vm::Eval>(c9).unwrap();
        assert_eq!(tape.len(), 9);
    }

    #[test]
    fn test_dupe() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let x_squared = ctx.mul(x, x).unwrap();

        let tape = ctx.get_tape::<crate::vm::Eval>(x_squared).unwrap();
        assert_eq!(tape.len(), 2);
    }

    #[test]
    fn test_var_errors() {
        let mut ctx = Context::new();
        assert!(ctx.var("X").is_err());

        let a1 = ctx.var("a").unwrap();
        let a2 = ctx.var("a").unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_remap_xyz() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let z = ctx.z();

        let s = ctx.add(x, 1.0).unwrap();

        let v = ctx.remap_xyz(s, [y, y, z]).unwrap();
        assert_eq!(ctx.eval_xyz(v, 0.0, 1.0, 0.0).unwrap(), 2.0);

        let one = ctx.constant(3.0);
        let v = ctx.remap_xyz(s, [one, y, z]).unwrap();
        assert_eq!(ctx.eval_xyz(v, 0.0, 1.0, 0.0).unwrap(), 4.0);
    }
}
