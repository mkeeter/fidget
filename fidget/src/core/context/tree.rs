//! Context-free math trees
use super::op::{BinaryOpcode, UnaryOpcode};
use std::sync::Arc;

/// Opcode type for trees
///
/// This is equivalent to [`Op`](crate::context::Op), but also includes the
/// [`RemapAxes`](TreeOp::RemapAxes) operation for lazy remapping.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum TreeOp {
    /// Input (at the moment, limited to "X", "Y", "Z")
    Input(&'static str),
    Const(f64),
    Binary(BinaryOpcode, Tree, Tree),
    Unary(UnaryOpcode, Tree),
    /// Lazy remapping of trees
    ///
    /// When imported into a `Context`, all `x/y/z` clauses within `target` will
    /// be replaced with the provided `x/y/z` trees.
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

impl PartialEq for Tree {
    /// Shallow (pointer) comparison
    ///
    /// This is implemented because `PartialEq` is required for
    /// [`nalgebra::Scalar`]; it's unlikely to be meaningful in user code.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.as_ptr(), other.as_ptr())
    }
}
impl Eq for Tree {}

/// See [`Context`](crate::Context) for documentation of these functions
#[allow(missing_docs)]
impl Tree {
    pub fn x() -> Self {
        Tree(Arc::new(TreeOp::Input("X")))
    }
    pub fn y() -> Self {
        Tree(Arc::new(TreeOp::Input("Y")))
    }
    pub fn z() -> Self {
        Tree(Arc::new(TreeOp::Input("Z")))
    }
    /// Returns an `(x, y, z)` tuple
    pub fn axes() -> (Self, Self, Self) {
        (Self::x(), Self::y(), Self::z())
    }
    pub fn constant(f: f64) -> Self {
        Tree(Arc::new(TreeOp::Const(f)))
    }
    fn op_unary(a: Tree, op: UnaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Unary(op, a)))
    }
    fn op_binary(a: Tree, b: Tree, op: BinaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Binary(op, a, b)))
    }
    pub fn square(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Square)
    }
    pub fn sqrt(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Sqrt)
    }
    pub fn max<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Max)
    }
    pub fn min<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Min)
    }
    pub fn neg(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Neg)
    }
    pub fn sin(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Sin)
    }
    pub fn cos(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Cos)
    }
    pub fn tan(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Tan)
    }
    pub fn asin(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Asin)
    }
    pub fn acos(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Acos)
    }
    pub fn atan(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Atan)
    }
    pub fn exp(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Exp)
    }
    pub fn ln(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Ln)
    }

    /// Returns a pointer to the inner [`TreeOp`]
    ///
    /// This can be used as a strong (but not unique) identity.
    pub fn as_ptr(&self) -> *const TreeOp {
        Arc::as_ptr(&self.0)
    }

    /// Remaps the axes of the given tree
    ///
    /// The remapping is lazy; it is not evaluated until the tree is imported
    /// into a `Context`.
    pub fn remap_xyz(&self, x: Tree, y: Tree, z: Tree) -> Tree {
        Self(Arc::new(TreeOp::RemapAxes {
            target: self.clone(),
            x,
            y,
            z,
        }))
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
        let x1 = ctx.import(&x1);
        let x2 = ctx.import(&x2);
        assert_eq!(x1, x2);
    }

    #[test]
    fn test_remap_xyz() {
        let s = Tree::x() + 1.0;

        let v = s.remap_xyz(Tree::y(), Tree::y(), Tree::z());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 2.0);

        let one = Tree::constant(3.0);
        let v = s.remap_xyz(one, Tree::y(), Tree::z());
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 4.0);
    }
}
