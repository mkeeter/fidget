//! Infrastructure for representing math expressions as trees and graphs
//!
//! There are two families of representations in this module:
//!
//! - A [`Tree`] is a free-floating math expression, which can be cloned
//!   and has overloaded operators for ease of use.  It is **not** deduplicated;
//!   two calls to [`Tree::constant(1.0)`](Tree::constant) will allocate two
//!   different objects.
//!   `Tree` objects are typically used when building up expressions; they
//!   should be converted to `Node` objects (in a particular `Context`) after
//!   they have been constructed.
//! - A [`Context`] is an arena for unique (deduplicated) math expressions,
//!   which are represented as [`Node`] handles.  Each `Node` is specific to a
//!   particular context.  Only `Node` objects can be converted into `Function`
//!   objects for evaluation.
//!
//! In other words, the typical workflow is `Tree → (Context, Node) → Function`.
mod indexed;
mod op;
mod tree;

use indexed::{define_index, Index, IndexMap, IndexVec};
pub use op::{BinaryOpcode, Op, UnaryOpcode};
pub use tree::{Tree, TreeOp};

use crate::{var::Var, Error};

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;

use ordered_float::OrderedFloat;

define_index!(Node, "An index in the `Context::ops` map");

/// A `Context` holds a set of deduplicated constants, variables, and
/// operations.
///
/// It should be used like an arena allocator: it grows over time, then frees
/// all of its contents when dropped.  There is no reference counting within the
/// context.
///
/// Items in the context are accessed with [`Node`] keys, which are simple
/// handles into an internal map.  Inside the context, operations are
/// represented with the [`Op`] type.
#[derive(Debug, Default)]
pub struct Context {
    ops: IndexMap<Op, Node>,
}

impl Context {
    /// Build a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the context
    ///
    /// All [`Node`] handles from this context are invalidated.
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

    /// Checks whether the given [`Node`] is valid in this context
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
    pub fn get_const(&self, n: Node) -> Result<f64, Error> {
        match self.get_op(n) {
            Some(Op::Const(c)) => Ok(c.0),
            Some(_) => Err(Error::NotAConst),
            _ => Err(Error::BadNode),
        }
    }

    /// Looks up the [`Var`] associated with the given node.
    ///
    /// If the node is invalid for this tree or not an `Op::Input`, returns an
    /// error.
    pub fn get_var(&self, n: Node) -> Result<Var, Error> {
        match self.get_op(n) {
            Some(Op::Input(v)) => Ok(*v),
            Some(..) => Err(Error::NotAVar),
            _ => Err(Error::BadNode),
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Primitives
    /// Constructs or finds a [`Var::X`] node
    /// ```
    /// # use fidget::context::Context;
    /// let mut ctx = Context::new();
    /// let x = ctx.x();
    /// let v = ctx.eval_xyz(x, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn x(&mut self) -> Node {
        self.var(Var::X)
    }

    /// Constructs or finds a [`Var::Y`] node
    pub fn y(&mut self) -> Node {
        self.var(Var::Y)
    }

    /// Constructs or finds a [`Var::Z`] node
    pub fn z(&mut self) -> Node {
        self.var(Var::Z)
    }

    /// Constructs or finds a variable input node
    ///
    /// To make an anonymous variable, call this function with [`Var::new()`]:
    ///
    /// ```
    /// # use fidget::{context::Context, var::Var};
    /// # use std::collections::HashMap;
    /// let mut ctx = Context::new();
    /// let v1 = ctx.var(Var::new());
    /// let v2 = ctx.var(Var::new());
    /// assert_ne!(v1, v2);
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert(ctx.get_var(v1).unwrap(), 3.0);
    /// assert_eq!(ctx.eval(v1, &vars).unwrap(), 3.0);
    /// assert!(ctx.eval(v2, &vars).is_err()); // v2 isn't in the map
    /// ```
    pub fn var(&mut self, v: Var) -> Node {
        self.ops.insert(Op::Input(v))
    }

    /// Returns a 3-element array of `X`, `Y`, `Z` nodes
    pub fn axes(&mut self) -> [Node; 3] {
        [self.x(), self.y(), self.z()]
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
            let v = self.eval(n, &Default::default())?;
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
            let v = self.eval(n, &Default::default())?;
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
            match (self.get_const(a), self.get_const(b)) {
                (Ok(zero), _) if zero == 0.0 => Ok(b),
                (_, Ok(zero)) if zero == 0.0 => Ok(a),
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
            match (self.get_const(a), self.get_const(b)) {
                (Ok(one), _) if one == 1.0 => Ok(b),
                (_, Ok(one)) if one == 1.0 => Ok(a),
                (Ok(zero), _) if zero == 0.0 => Ok(a),
                (_, Ok(zero)) if zero == 0.0 => Ok(b),
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

    /// Builds an `and` node
    ///
    /// If both arguments are non-zero, returns the right-hand argument.
    /// Otherwise, returns zero.
    ///
    /// This node can be simplified using a tracing evaluator:
    /// - If the left-hand argument is zero, simplify to just that argument
    /// - If the left-hand argument is non-zero, simplify to the other argument
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.and(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// let v = ctx.eval_xyz(op, 1.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 1.0, 2.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn and<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;

        let op_a = *self.get_op(a).ok_or(Error::BadNode)?;
        if let Op::Const(v) = op_a {
            if v.0 == 0.0 {
                Ok(a)
            } else {
                Ok(b)
            }
        } else {
            self.op_binary(a, b, BinaryOpcode::And)
        }
    }

    /// Builds an `or` node
    ///
    /// If the left-hand argument is non-zero, it is returned.  Otherwise, the
    /// right-hand argument is returned.
    ///
    /// This node can be simplified using a tracing evaluator.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.or(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 0.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// let v = ctx.eval_xyz(op, 0.0, 3.0, 0.0).unwrap();
    /// assert_eq!(v, 3.0);
    /// ```
    pub fn or<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;

        let op_a = *self.get_op(a).ok_or(Error::BadNode)?;
        let op_b = *self.get_op(b).ok_or(Error::BadNode)?;
        if let Op::Const(v) = op_a {
            if v.0 != 0.0 {
                return Ok(a);
            } else {
                return Ok(b);
            }
        } else if let Op::Const(v) = op_b {
            if v.0 == 0.0 {
                return Ok(a);
            }
        }
        self.op_binary(a, b, BinaryOpcode::Or)
    }

    /// Builds a logical negation node
    ///
    /// The output is 1 if the argument is 0, and 0 otherwise.
    pub fn not<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Not)
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

    /// Builds a node which calculates the sine of its input (in radians)
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.sin(x).unwrap();
    /// let v = ctx.eval_xyz(op, std::f64::consts::PI / 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn sin<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Sin)
    }

    /// Builds a node which calculates the cosine of its input (in radians)
    pub fn cos<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Cos)
    }

    /// Builds a node which calculates the tangent of its input (in radians)
    pub fn tan<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Tan)
    }

    /// Builds a node which calculates the arcsine of its input (in radians)
    pub fn asin<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Asin)
    }

    /// Builds a node which calculates the arccosine of its input (in radians)
    pub fn acos<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Acos)
    }

    /// Builds a node which calculates the arctangent of its input (in radians)
    pub fn atan<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Atan)
    }

    /// Builds a node which calculates the exponent of its input
    pub fn exp<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Exp)
    }

    /// Builds a node which calculates the natural log of its input
    pub fn ln<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Ln)
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

    /// Builds a node which takes the floor of its input
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.floor(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.2, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// ```
    pub fn floor<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Floor)
    }

    /// Builds a node which takes the ceiling of its input
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.ceil(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.2, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// ```
    pub fn ceil<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Ceil)
    }

    /// Builds a node which rounds its input to the nearest integer
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.round(x).unwrap();
    /// let v = ctx.eval_xyz(op, 1.2, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 1.6, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0);
    /// let v = ctx.eval_xyz(op, 1.5, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 2.0); // rounds away from 0.0 if ambiguous
    /// ```
    pub fn round<A: IntoNode>(&mut self, a: A) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        self.op_unary(a, UnaryOpcode::Round)
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

        match (self.get_const(a), self.get_const(b)) {
            (Ok(zero), _) if zero == 0.0 => self.neg(b),
            (_, Ok(zero)) if zero == 0.0 => Ok(a),
            _ => self.op_binary(a, b, BinaryOpcode::Sub),
        }
    }

    /// Builds a node which performs division.
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

        match (self.get_const(a), self.get_const(b)) {
            (Ok(zero), _) if zero == 0.0 => Ok(a),
            (_, Ok(one)) if one == 1.0 => Ok(a),
            _ => self.op_binary(a, b, BinaryOpcode::Div),
        }
    }

    /// Builds a node which computes `atan2(y, x)`
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.atan2(y, x).unwrap();
    /// let v = ctx.eval_xyz(op, 0.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, std::f64::consts::FRAC_PI_2);
    /// ```
    pub fn atan2<A: IntoNode, B: IntoNode>(
        &mut self,
        y: A,
        x: B,
    ) -> Result<Node, Error> {
        let y = y.into_node(self)?;
        let x = x.into_node(self)?;

        self.op_binary(y, x, BinaryOpcode::Atan)
    }

    /// Builds a node that compares two values
    ///
    /// The result is -1 if `a < b`, +1 if `a > b`, 0 if `a == b`, and `NaN` if
    /// either side is `NaN`.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let op = ctx.compare(x, 1.0).unwrap();
    /// let v = ctx.eval_xyz(op, 0.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, -1.0);
    /// let v = ctx.eval_xyz(op, 2.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 1.0, 0.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// ```
    pub fn compare<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;
        self.op_binary(a, b, BinaryOpcode::Compare)
    }

    /// Builds a node that is 1 if `lhs < rhs` and 0 otherwise
    ///
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.less_than(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 0.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 1.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// let v = ctx.eval_xyz(op, 2.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// ```
    pub fn less_than<A: IntoNode, B: IntoNode>(
        &mut self,
        lhs: A,
        rhs: B,
    ) -> Result<Node, Error> {
        let lhs = lhs.into_node(self)?;
        let rhs = rhs.into_node(self)?;
        let cmp = self.op_binary(rhs, lhs, BinaryOpcode::Compare)?;
        self.max(cmp, 0.0)
    }

    /// Builds a node that is 1 if `lhs <= rhs` and 0 otherwise
    ///
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let op = ctx.less_than(x, y).unwrap();
    /// let v = ctx.eval_xyz(op, 0.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 1.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 1.0);
    /// let v = ctx.eval_xyz(op, 2.0, 1.0, 0.0).unwrap();
    /// assert_eq!(v, 0.0);
    /// ```
    pub fn less_than_or_equal<A: IntoNode, B: IntoNode>(
        &mut self,
        lhs: A,
        rhs: B,
    ) -> Result<Node, Error> {
        let lhs = lhs.into_node(self)?;
        let rhs = rhs.into_node(self)?;
        let cmp = self.op_binary(rhs, lhs, BinaryOpcode::Compare)?;
        let shift = self.add(cmp, 1.0)?;
        self.div(shift, 2.0)
    }

    /// Builds a node that takes the modulo (least non-negative remainder)
    pub fn modulo<A: IntoNode, B: IntoNode>(
        &mut self,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;
        self.op_binary(a, b, BinaryOpcode::Mod)
    }

    /// Builds a node that returns the first node if the condition is not
    /// equal to zero, else returns the other node
    ///
    /// The result is `a` if `condition != 0`, else the result is `b`.
    /// ```
    /// # let mut ctx = fidget::context::Context::new();
    /// let x = ctx.x();
    /// let y = ctx.y();
    /// let z = ctx.z();
    ///
    /// let if_else = ctx.if_nonzero_else(x, y, z).unwrap();
    ///
    /// assert_eq!(ctx.eval_xyz(if_else, 0.0, 2.0, 3.0).unwrap(), 3.0);
    /// assert_eq!(ctx.eval_xyz(if_else, 1.0, 2.0, 3.0).unwrap(), 2.0);
    /// assert_eq!(ctx.eval_xyz(if_else, 0.0, f64::NAN, 3.0).unwrap(), 3.0);
    /// assert_eq!(ctx.eval_xyz(if_else, 1.0, 2.0, f64::NAN).unwrap(), 2.0);
    /// ```
    pub fn if_nonzero_else<Condition: IntoNode, A: IntoNode, B: IntoNode>(
        &mut self,
        condition: Condition,
        a: A,
        b: B,
    ) -> Result<Node, Error> {
        let condition = condition.into_node(self)?;
        let a = a.into_node(self)?;
        let b = b.into_node(self)?;

        let lhs = self.and(condition, a)?;
        let n_condition = self.not(condition)?;
        let rhs = self.and(n_condition, b)?;
        self.or(lhs, rhs)
    }

    ////////////////////////////////////////////////////////////////////////////
    /// Evaluates the given node with the provided values for X, Y, and Z.
    ///
    /// This is extremely inefficient; consider converting the node into a
    /// [`Shape`](crate::shape::Shape) and using its evaluators instead.
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
        let vars = [(Var::X, x), (Var::Y, y), (Var::Z, z)]
            .into_iter()
            .collect();
        self.eval(root, &vars)
    }

    /// Evaluates the given node with a generic set of variables
    ///
    /// This is extremely inefficient; consider converting the node into a
    /// [`Shape`](crate::shape::Shape) and using its evaluators instead.
    pub fn eval(
        &self,
        root: Node,
        vars: &HashMap<Var, f64>,
    ) -> Result<f64, Error> {
        let mut cache = vec![None; self.ops.len()].into();
        self.eval_inner(root, vars, &mut cache)
    }

    fn eval_inner(
        &self,
        node: Node,
        vars: &HashMap<Var, f64>,
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
            Op::Input(v) => *vars.get(v).ok_or(Error::MissingVar(*v))?,
            Op::Const(c) => c.0,

            Op::Binary(op, a, b) => {
                let a = get(*a)?;
                let b = get(*b)?;
                match op {
                    BinaryOpcode::Add => a + b,
                    BinaryOpcode::Sub => a - b,
                    BinaryOpcode::Mul => a * b,
                    BinaryOpcode::Div => a / b,
                    BinaryOpcode::Atan => a.atan2(b),
                    BinaryOpcode::Min => a.min(b),
                    BinaryOpcode::Max => a.max(b),
                    BinaryOpcode::Compare => a
                        .partial_cmp(&b)
                        .map(|i| i as i8 as f64)
                        .unwrap_or(f64::NAN),
                    BinaryOpcode::Mod => a.rem_euclid(b),
                    BinaryOpcode::And => {
                        if a == 0.0 {
                            a
                        } else {
                            b
                        }
                    }
                    BinaryOpcode::Or => {
                        if a != 0.0 {
                            a
                        } else {
                            b
                        }
                    }
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
                    UnaryOpcode::Not => (a == 0.0).into(),
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
                "floor" => ctx.floor(pop()?)?,
                "ceil" => ctx.ceil(pop()?)?,
                "round" => ctx.round(pop()?)?,
                "sin" => ctx.sin(pop()?)?,
                "cos" => ctx.cos(pop()?)?,
                "tan" => ctx.tan(pop()?)?,
                "asin" => ctx.asin(pop()?)?,
                "acos" => ctx.acos(pop()?)?,
                "atan" => ctx.atan(pop()?)?,
                "ln" => ctx.ln(pop()?)?,
                "not" => ctx.not(pop()?)?,
                "exp" => ctx.exp(pop()?)?,
                "add" => ctx.add(pop()?, pop()?)?,
                "mul" => ctx.mul(pop()?, pop()?)?,
                "min" => ctx.min(pop()?, pop()?)?,
                "max" => ctx.max(pop()?, pop()?)?,
                "div" => ctx.div(pop()?, pop()?)?,
                "atan2" => ctx.atan2(pop()?, pop()?)?,
                "sub" => ctx.sub(pop()?, pop()?)?,
                "compare" => ctx.compare(pop()?, pop()?)?,
                "mod" => ctx.modulo(pop()?, pop()?)?,
                "and" => ctx.and(pop()?, pop()?)?,
                "or" => ctx.or(pop()?, pop()?)?,
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
    fn dot_node(&self, i: Node) -> String {
        let mut out = format!(r#"n{} [label = ""#, i.get());
        let op = self.get_op(i).unwrap();
        match op {
            Op::Const(c) => write!(out, "{}", c).unwrap(),
            Op::Input(v) => {
                out += &v.to_string();
            }
            Op::Binary(op, ..) => match op {
                BinaryOpcode::Add => out += "add",
                BinaryOpcode::Sub => out += "sub",
                BinaryOpcode::Mul => out += "mul",
                BinaryOpcode::Div => out += "div",
                BinaryOpcode::Atan => out += "atan2",
                BinaryOpcode::Min => out += "min",
                BinaryOpcode::Max => out += "max",
                BinaryOpcode::Compare => out += "compare",
                BinaryOpcode::Mod => out += "mod",
                BinaryOpcode::And => out += "and",
                BinaryOpcode::Or => out += "or",
            },
            Op::Unary(op, ..) => match op {
                UnaryOpcode::Neg => out += "neg",
                UnaryOpcode::Abs => out += "abs",
                UnaryOpcode::Recip => out += "recip",
                UnaryOpcode::Sqrt => out += "sqrt",
                UnaryOpcode::Square => out += "square",
                UnaryOpcode::Floor => out += "floor",
                UnaryOpcode::Ceil => out += "ceil",
                UnaryOpcode::Round => out += "round",
                UnaryOpcode::Sin => out += "sin",
                UnaryOpcode::Cos => out += "cos",
                UnaryOpcode::Tan => out += "tan",
                UnaryOpcode::Asin => out += "asin",
                UnaryOpcode::Acos => out += "acos",
                UnaryOpcode::Atan => out += "atan",
                UnaryOpcode::Exp => out += "exp",
                UnaryOpcode::Ln => out += "ln",
                UnaryOpcode::Not => out += "not",
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

    /// Imports the given tree, deduplicating and returning the root
    pub fn import(&mut self, tree: &Tree) -> Node {
        // A naive remapping implementation would use recursion.  A naive
        // remapping implementation would blow up the stack given any
        // significant tree size.
        //
        // Instead, we maintain our own pseudo-stack here in a pair of Vecs (one
        // stack for actions, and a second stack for return values).
        enum Action<'a> {
            /// Pushes `Up(op)` followed by `Down(c)` for each child
            Down(&'a Arc<TreeOp>),
            /// Consumes imported trees from the stack and pushes a new tree
            Up(&'a Arc<TreeOp>),
            /// Pops the latest axis frame
            Pop,
        }
        let mut axes = vec![(self.x(), self.y(), self.z())];
        let mut todo = vec![Action::Down(tree.arc())];
        let mut stack = vec![];

        // Cache of TreeOp -> Node mapping under a particular frame (axes)
        //
        // This isn't required for correctness, but can be a speed optimization
        // (because it means we don't have to walk the same tree twice).
        let mut seen = HashMap::new();

        while let Some(t) = todo.pop() {
            match t {
                Action::Down(t) => {
                    // If we've already seen this TreeOp with these axes, then
                    // we can return the previous Node.
                    if matches!(
                        t.as_ref(),
                        TreeOp::Unary(..) | TreeOp::Binary(..)
                    ) {
                        if let Some(p) =
                            seen.get(&(*axes.last().unwrap(), Arc::as_ptr(t)))
                        {
                            stack.push(*p);
                            continue;
                        }
                    }
                    match t.as_ref() {
                        TreeOp::Const(c) => {
                            stack.push(self.constant(*c));
                        }
                        TreeOp::Input(s) => {
                            let axes = axes.last().unwrap();
                            stack.push(match *s {
                                Var::X => axes.0,
                                Var::Y => axes.1,
                                Var::Z => axes.2,
                                v @ Var::V(..) => self.var(v),
                            });
                        }
                        TreeOp::Unary(_op, arg) => {
                            todo.push(Action::Up(t));
                            todo.push(Action::Down(arg));
                        }
                        TreeOp::Binary(_op, lhs, rhs) => {
                            todo.push(Action::Up(t));
                            todo.push(Action::Down(lhs));
                            todo.push(Action::Down(rhs));
                        }
                        TreeOp::RemapAxes { target: _, x, y, z } => {
                            // Action::Up(t) does the remapping and target eval
                            todo.push(Action::Up(t));
                            todo.push(Action::Down(x));
                            todo.push(Action::Down(y));
                            todo.push(Action::Down(z));
                        }
                    }
                }
                Action::Up(t) => {
                    match t.as_ref() {
                        TreeOp::Const(..) | TreeOp::Input(..) => unreachable!(),
                        TreeOp::Unary(op, ..) => {
                            let arg = stack.pop().unwrap();
                            let out = self.op_unary(arg, *op).unwrap();
                            stack.push(out);
                        }
                        TreeOp::Binary(op, ..) => {
                            let lhs = stack.pop().unwrap();
                            let rhs = stack.pop().unwrap();
                            let out = self.op_binary(lhs, rhs, *op).unwrap();
                            if Arc::strong_count(t) > 1 {
                                seen.insert(
                                    (*axes.last().unwrap(), Arc::as_ptr(t)),
                                    out,
                                );
                            }
                            stack.push(out);
                        }
                        TreeOp::RemapAxes { target, .. } => {
                            let x = stack.pop().unwrap();
                            let y = stack.pop().unwrap();
                            let z = stack.pop().unwrap();
                            axes.push((x, y, z));
                            todo.push(Action::Pop);
                            todo.push(Action::Down(target));
                        }
                    }
                    // Update the cache with the new tree, if relevant
                    //
                    // The `strong_count` check is a rough heuristic to avoid
                    // caching if there's only a single owner of the tree.  This
                    // isn't perfect, but it doesn't need to be for correctness.
                    if matches!(
                        t.as_ref(),
                        TreeOp::Unary(..) | TreeOp::Binary(..)
                    ) && Arc::strong_count(t) > 1
                    {
                        seen.insert(
                            (*axes.last().unwrap(), Arc::as_ptr(t)),
                            *stack.last().unwrap(),
                        );
                    }
                }
                Action::Pop => {
                    axes.pop().unwrap();
                }
            }
        }
        assert_eq!(stack.len(), 1);
        stack.pop().unwrap()
    }

    /// Converts from a context-specific node into a standalone [`Tree`]
    pub fn export(&self, n: Node) -> Result<Tree, Error> {
        if self.get_op(n).is_none() {
            return Err(Error::BadNode);
        }

        // Do recursion on the heap to avoid stack overflows for deep trees
        enum Action {
            /// Pushes `Up(n)` followed by `Down(n)` for each child
            Down(Node),
            /// Consumes trees from the stack and pushes a new tree
            Up(Node, Op),
        }
        let mut todo = vec![Action::Down(n)];
        let mut stack = vec![];

        // Cache of Node -> Tree mapping, for Tree deduplication
        let mut seen: HashMap<Node, Tree> = HashMap::new();

        while let Some(t) = todo.pop() {
            match t {
                Action::Down(n) => {
                    // If we've already seen this TreeOp with these axes, then
                    // we can return the previous Node.
                    if let Some(p) = seen.get(&n) {
                        stack.push(p.clone());
                        continue;
                    }
                    let op = self.get_op(n).unwrap();
                    match op {
                        Op::Const(c) => {
                            let t = Tree::from(c.0);
                            seen.insert(n, t.clone());
                            stack.push(t);
                        }
                        Op::Input(v) => {
                            let t = Tree::from(*v);
                            seen.insert(n, t.clone());
                            stack.push(t);
                        }
                        Op::Unary(_op, arg) => {
                            todo.push(Action::Up(n, *op));
                            todo.push(Action::Down(*arg));
                        }
                        Op::Binary(_op, lhs, rhs) => {
                            todo.push(Action::Up(n, *op));
                            todo.push(Action::Down(*lhs));
                            todo.push(Action::Down(*rhs));
                        }
                    }
                }
                Action::Up(n, op) => match op {
                    Op::Const(..) | Op::Input(..) => unreachable!(),
                    Op::Unary(op, ..) => {
                        let arg = stack.pop().unwrap();
                        let out =
                            Tree::from(TreeOp::Unary(op, arg.arc().clone()));
                        seen.insert(n, out.clone());
                        stack.push(out);
                    }
                    Op::Binary(op, ..) => {
                        let lhs = stack.pop().unwrap();
                        let rhs = stack.pop().unwrap();
                        let out = Tree::from(TreeOp::Binary(
                            op,
                            lhs.arc().clone(),
                            rhs.arc().clone(),
                        ));
                        seen.insert(n, out.clone());
                        stack.push(out);
                    }
                },
            }
        }
        assert_eq!(stack.len(), 1);
        Ok(stack.pop().unwrap())
    }

    /// Takes the symbolic derivative of a node with respect to a variable
    pub fn deriv(&mut self, n: Node, v: Var) -> Result<Node, Error> {
        if self.get_op(n).is_none() {
            return Err(Error::BadNode);
        }

        // Do recursion on the heap to avoid stack overflows for deep trees
        enum Action {
            /// Pushes `Up(n)` followed by `Down(n)` for each child
            Down(Node),
            /// Consumes trees from the stack and pushes a new tree
            Up(Node, Op),
        }
        let mut todo = vec![Action::Down(n)];
        let mut stack = vec![];
        let zero = self.constant(0.0);

        // Cache of Node -> Node mapping, for deduplication
        let mut seen: HashMap<Node, Node> = HashMap::new();

        while let Some(t) = todo.pop() {
            match t {
                Action::Down(n) => {
                    // If we've already seen this TreeOp with these axes, then
                    // we can return the previous Node.
                    if let Some(p) = seen.get(&n) {
                        stack.push(*p);
                        continue;
                    }
                    let op = *self.get_op(n).unwrap();
                    match op {
                        Op::Const(_c) => {
                            seen.insert(n, zero);
                            stack.push(zero);
                        }
                        Op::Input(u) => {
                            let z =
                                if v == u { self.constant(1.0) } else { zero };
                            seen.insert(n, z);
                            stack.push(z);
                        }
                        Op::Unary(_op, arg) => {
                            todo.push(Action::Up(n, op));
                            todo.push(Action::Down(arg));
                        }
                        Op::Binary(_op, lhs, rhs) => {
                            todo.push(Action::Up(n, op));
                            todo.push(Action::Down(lhs));
                            todo.push(Action::Down(rhs));
                        }
                    }
                }
                Action::Up(n, op) => match op {
                    Op::Const(..) | Op::Input(..) => unreachable!(),
                    Op::Unary(op, v_arg) => {
                        let d_arg = stack.pop().unwrap();
                        let out = match op {
                            UnaryOpcode::Neg => self.neg(d_arg),
                            UnaryOpcode::Abs => {
                                let cond = self.less_than(v_arg, zero).unwrap();
                                let pos = d_arg;
                                let neg = self.neg(d_arg).unwrap();
                                self.if_nonzero_else(cond, neg, pos)
                            }
                            UnaryOpcode::Recip => {
                                let a = self.square(v_arg).unwrap();
                                let b = self.neg(d_arg).unwrap();
                                self.div(b, a)
                            }
                            UnaryOpcode::Sqrt => {
                                let v = self.mul(n, 2.0).unwrap();
                                self.div(d_arg, v)
                            }
                            UnaryOpcode::Square => {
                                let v = self.mul(d_arg, v_arg).unwrap();
                                self.mul(2.0, v)
                            }
                            // Discontinuous constants don't have Dirac deltas
                            UnaryOpcode::Floor
                            | UnaryOpcode::Ceil
                            | UnaryOpcode::Round => Ok(zero),

                            UnaryOpcode::Sin => {
                                let c = self.cos(v_arg).unwrap();
                                self.mul(c, d_arg)
                            }

                            UnaryOpcode::Cos => {
                                let s = self.sin(v_arg).unwrap();
                                let s = self.neg(s).unwrap();
                                self.mul(s, d_arg)
                            }

                            UnaryOpcode::Tan => {
                                let c = self.cos(v_arg).unwrap();
                                let c = self.square(c).unwrap();
                                self.div(d_arg, c)
                            }

                            UnaryOpcode::Asin => {
                                let v = self.square(v_arg).unwrap();
                                let v = self.sub(1.0, v).unwrap();
                                let v = self.sqrt(v).unwrap();
                                self.div(d_arg, v)
                            }
                            UnaryOpcode::Acos => {
                                let v = self.square(v_arg).unwrap();
                                let v = self.sub(1.0, v).unwrap();
                                let v = self.sqrt(v).unwrap();
                                let v = self.neg(v).unwrap();
                                self.div(d_arg, v)
                            }
                            UnaryOpcode::Atan => {
                                let v = self.square(v_arg).unwrap();
                                let v = self.add(1.0, v).unwrap();
                                self.div(d_arg, v)
                            }
                            UnaryOpcode::Exp => self.mul(n, d_arg),
                            UnaryOpcode::Ln => self.div(d_arg, v_arg),
                            UnaryOpcode::Not => Ok(zero),
                        }
                        .unwrap();
                        seen.insert(n, out);
                        stack.push(out);
                    }
                    Op::Binary(op, v_lhs, v_rhs) => {
                        let d_lhs = stack.pop().unwrap();
                        let d_rhs = stack.pop().unwrap();
                        let out = match op {
                            BinaryOpcode::Add => self.add(d_lhs, d_rhs),
                            BinaryOpcode::Sub => self.sub(d_lhs, d_rhs),
                            BinaryOpcode::Mul => {
                                let a = self.mul(d_lhs, v_rhs).unwrap();
                                let b = self.mul(v_lhs, d_rhs).unwrap();
                                self.add(a, b)
                            }
                            BinaryOpcode::Div => {
                                let v = self.square(v_rhs).unwrap();
                                let a = self.mul(v_rhs, d_lhs).unwrap();
                                let b = self.mul(v_lhs, d_rhs).unwrap();
                                let c = self.sub(a, b).unwrap();
                                self.div(c, v)
                            }
                            BinaryOpcode::Atan => {
                                let a = self.square(v_rhs).unwrap();
                                let b = self.square(v_rhs).unwrap();
                                let d = self.add(a, b).unwrap();

                                let a = self.mul(v_rhs, d_lhs).unwrap();
                                let b = self.mul(v_lhs, d_rhs).unwrap();
                                let v = self.sub(a, b).unwrap();
                                self.div(v, d)
                            }
                            BinaryOpcode::Min => {
                                let cond =
                                    self.less_than(v_lhs, v_rhs).unwrap();
                                self.if_nonzero_else(cond, d_lhs, d_rhs)
                            }
                            BinaryOpcode::Max => {
                                let cond =
                                    self.less_than(v_rhs, v_lhs).unwrap();
                                self.if_nonzero_else(cond, d_lhs, d_rhs)
                            }
                            BinaryOpcode::Compare => Ok(zero),
                            BinaryOpcode::Mod => {
                                let e = self.div(v_lhs, v_rhs).unwrap();
                                let q = self.floor(e).unwrap();

                                // XXX
                                // (we don't actually have %, so hack it from
                                // `modulo`, which is actually `rem_euclid`)
                                // ???
                                let m = self.modulo(q, v_rhs).unwrap();
                                let cond = self.less_than(q, zero).unwrap();
                                let offset = self
                                    .if_nonzero_else(cond, v_rhs, zero)
                                    .unwrap();
                                let m = self.sub(m, offset).unwrap();

                                // Torn from the div_euclid implementation
                                let outer = self.less_than(m, zero).unwrap();
                                let inner =
                                    self.less_than(zero, v_rhs).unwrap();
                                let qa = self.sub(q, 1.0).unwrap();
                                let qb = self.add(q, 1.0).unwrap();
                                let inner = self
                                    .if_nonzero_else(inner, qa, qb)
                                    .unwrap();
                                let e = self
                                    .if_nonzero_else(outer, inner, q)
                                    .unwrap();

                                let v = self.mul(d_rhs, e).unwrap();
                                self.sub(d_lhs, v)
                            }
                            BinaryOpcode::And => {
                                let cond = self.compare(v_lhs, zero).unwrap();
                                self.if_nonzero_else(cond, d_rhs, d_lhs)
                            }
                            BinaryOpcode::Or => {
                                let cond = self.compare(v_lhs, zero).unwrap();
                                self.if_nonzero_else(cond, d_lhs, d_rhs)
                            }
                        }
                        .unwrap();
                        seen.insert(n, out);
                        stack.push(out);
                    }
                },
            }
        }
        assert_eq!(stack.len(), 1);
        Ok(stack.pop().unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Helper trait for things that can be converted into a [`Node`] given a
/// [`Context`].
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
    use crate::vm::VmData;

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

        let tape = VmData::<255>::new(&ctx, c9).unwrap();
        assert_eq!(tape.len(), 8);
        assert_eq!(tape.vars.len(), 2);
    }

    #[test]
    fn test_dupe() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let x_squared = ctx.mul(x, x).unwrap();

        let tape = VmData::<255>::new(&ctx, x_squared).unwrap();
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.vars.len(), 1);
    }

    #[test]
    fn test_export() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sin(x).unwrap();
        let c = ctx.cos(x).unwrap();
        let sum = ctx.add(s, c).unwrap();
        let t = ctx.export(sum).unwrap();
        if let TreeOp::Binary(BinaryOpcode::Add, lhs, rhs) = &*t {
            match (&**lhs, &**rhs) {
                (
                    TreeOp::Unary(UnaryOpcode::Sin, x1),
                    TreeOp::Unary(UnaryOpcode::Cos, x2),
                ) => {
                    assert_eq!(Arc::as_ptr(x1), Arc::as_ptr(x2));
                    let TreeOp::Input(Var::X) = &**x1 else {
                        panic!("invalid X: {x1:?}");
                    };
                }
                _ => panic!("invalid lhs / rhs: {lhs:?} {rhs:?}"),
            }
        } else {
            panic!("unexpected opcode {t:?}");
        }
    }
}
