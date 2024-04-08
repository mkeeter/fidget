//! Context-free math trees
//!
//! The [`Tree`] and [`TreeOp`]
use super::op::{BinaryOpcode, UnaryOpcode};
use std::sync::Arc;

/// Opcode type for trees
///
/// This is equivalent to [`Op`](crate::context::Op), but also includes the
/// [`RemapAxes`](TreeOp::RemapAxes) operation for lazy remapping.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum TreeOp {
    Input(&'static str),
    Const(f64),
    Binary(BinaryOpcode, Tree, Tree),
    Unary(UnaryOpcode, Tree),
    RemapAxes {
        target: Tree,
        x: Tree,
        y: Tree,
        z: Tree,
    },
}

impl From<f64> for Tree {
    fn from(v: f64) -> Tree {
        Tree::constant(v)
    }
}

impl From<f32> for Tree {
    fn from(v: f32) -> Tree {
        Tree::constant(v as f64)
    }
}

/// Owned handle for a standalone math tree
#[derive(Clone, Debug)]
pub struct Tree(Arc<TreeOp>);

impl std::ops::Deref for Tree {
    type Target = TreeOp;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// XXX `PartialEq` is required for `nalgebra`, so we'll do pointer equality
impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.as_ref() as *const _, other.0.as_ref() as *const _)
    }
}
impl Eq for Tree {}

impl Tree {
    /// Builds a tree that evaluates the `x` coordinate
    pub fn x() -> Self {
        Tree(Arc::new(TreeOp::Input("X")))
    }
    /// Builds a tree that evaluates the `y` coordinate
    pub fn y() -> Self {
        Tree(Arc::new(TreeOp::Input("Y")))
    }
    /// Builds a tree that evaluates the `z` coordinate
    pub fn z() -> Self {
        Tree(Arc::new(TreeOp::Input("Z")))
    }
    /// Returns an `(x, y, z)` tuple
    pub fn axes() -> (Self, Self, Self) {
        (Self::x(), Self::y(), Self::z())
    }
    /// Builds a tree from the given constant
    pub fn constant(f: f64) -> Self {
        Tree(Arc::new(TreeOp::Const(f)))
    }
    fn op_unary(a: Tree, op: UnaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Unary(op, a)))
    }
    fn op_binary(a: Tree, b: Tree, op: BinaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Binary(op, a, b)))
    }

    /// Returns the square of this tree
    pub fn square(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Square)
    }

    /// Returns the square of this tree
    pub fn sqrt(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Sqrt)
    }

    /// Takes the maximum of two trees
    pub fn max<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Max)
    }

    /// Takes the maximum of two trees
    pub fn min<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Min)
    }

    /// Returns a pointer to the inner [`TreeOp`]
    ///
    /// This can be used as a strong (but not unique) identity.
    pub fn as_ptr(&self) -> *const TreeOp {
        Arc::as_ptr(&self.0)
    }
}

macro_rules! impl_binary {
    ($op:ident, $op_assign:ident, $base_fn:ident, $assign_fn:ident) => {
        impl<A: Into<Tree>> std::ops::$op<A> for Tree {
            type Output = Self;

            fn $base_fn(self, other: A) -> Self {
                Self::op_binary(self, other.into(), BinaryOpcode::$op)
            }
        }
        impl<A: Into<Tree>> std::ops::$op_assign<A> for Tree {
            fn $assign_fn(&mut self, other: A) {
                use std::ops::$op;
                self.0 = self.clone().$base_fn(other.into()).0
            }
        }
        impl std::ops::$op<Tree> for f32 {
            type Output = Tree;
            fn $base_fn(self, other: Tree) -> Tree {
                Tree::op_binary(self.into(), other, BinaryOpcode::$op)
            }
        }
        impl std::ops::$op<Tree> for f64 {
            type Output = Tree;
            fn $base_fn(self, other: Tree) -> Tree {
                Tree::op_binary(self.into(), other, BinaryOpcode::$op)
            }
        }
    };
}

impl_binary!(Add, AddAssign, add, add_assign);
impl_binary!(Sub, SubAssign, sub, sub_assign);
impl_binary!(Mul, MulAssign, mul, mul_assign);
impl_binary!(Div, DivAssign, div, div_assign);

#[cfg(test)]
mod test {
    use super::*;
    use crate::Context;

    #[test]
    fn tree_x() {
        let x1 = Tree::x();
        let x2 = Tree::x();
        assert_ne!(x1, x2);

        let mut ctx = Context::new();
        let x1 = ctx.import(x1);
        let x2 = ctx.import(x2);
        assert_eq!(x1, x2);
    }
}
