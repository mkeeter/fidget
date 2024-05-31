//! Context-free math trees
use super::op::{BinaryOpcode, UnaryOpcode};
use crate::var::Var;
use std::sync::Arc;

/// Opcode type for trees
///
/// This is equivalent to [`Op`](crate::context::Op), but also includes the
/// [`RemapAxes`](TreeOp::RemapAxes) operation for lazy remapping.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum TreeOp {
    /// Input (an arbitrary [`Var`])
    Input(Var),
    Const(f64),
    Binary(BinaryOpcode, Arc<TreeOp>, Arc<TreeOp>),
    Unary(UnaryOpcode, Arc<TreeOp>),
    /// Lazy remapping of trees
    ///
    /// When imported into a `Context`, all `x/y/z` clauses within `target` will
    /// be replaced with the provided `x/y/z` trees.
    RemapAxes {
        target: Arc<TreeOp>,
        x: Arc<TreeOp>,
        y: Arc<TreeOp>,
        z: Arc<TreeOp>,
    },
}

impl Drop for TreeOp {
    fn drop(&mut self) {
        // Early exit for TreeOps which have limited recursion
        if self.fast_drop() {
            return;
        }

        let mut todo = vec![std::mem::replace(self, TreeOp::Const(0.0))];
        let empty = Arc::new(TreeOp::Const(0.0));
        while let Some(mut t) = todo.pop() {
            for t in t.iter_children() {
                let arg = std::mem::replace(t, empty.clone());
                todo.extend(Arc::into_inner(arg));
            }
            drop(t);
        }
    }
}

impl TreeOp {
    /// Checks whether the given tree is eligible for fast dropping
    ///
    /// Fast dropping uses the normal `Drop` implementation, which recurses on
    /// the stack and can overflow for deep trees.  A recursive tree is only
    /// eligible for fast dropping if all of its children are `TreeOp::Const`.
    fn fast_drop(&self) -> bool {
        match self {
            TreeOp::Const(..) | TreeOp::Input(..) => true,
            TreeOp::Unary(_op, arg) => matches!(**arg, TreeOp::Const(..)),
            TreeOp::Binary(_op, lhs, rhs) => {
                matches!(**lhs, TreeOp::Const(..))
                    && matches!(**rhs, TreeOp::Const(..))
            }
            TreeOp::RemapAxes { target, x, y, z } => {
                matches!(**target, TreeOp::Const(..))
                    && matches!(**x, TreeOp::Const(..))
                    && matches!(**y, TreeOp::Const(..))
                    && matches!(**z, TreeOp::Const(..))
            }
        }
    }

    fn iter_children(&mut self) -> impl Iterator<Item = &mut Arc<TreeOp>> {
        match self {
            TreeOp::Const(..) | TreeOp::Input(..) => [None, None, None, None],
            TreeOp::Unary(_op, arg) => [Some(arg), None, None, None],
            TreeOp::Binary(_op, lhs, rhs) => [Some(lhs), Some(rhs), None, None],
            TreeOp::RemapAxes { target, x, y, z } => {
                [Some(target), Some(x), Some(y), Some(z)]
            }
        }
        .into_iter()
        .flatten()
    }
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

impl From<i32> for Tree {
    fn from(v: i32) -> Tree {
        Tree::constant(v as f64)
    }
}

impl From<Var> for Tree {
    fn from(v: Var) -> Tree {
        Tree(Arc::new(TreeOp::Input(v)))
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

impl Tree {
    /// Returns an `(x, y, z)` tuple
    pub fn axes() -> (Self, Self, Self) {
        (Self::x(), Self::y(), Self::z())
    }

    /// Returns a pointer to the inner [`TreeOp`]
    ///
    /// This can be used as a strong (but not unique) identity.
    pub fn as_ptr(&self) -> *const TreeOp {
        Arc::as_ptr(&self.0)
    }

    /// Borrow the inner `Arc<TreeOp>`
    pub(crate) fn arc(&self) -> &Arc<TreeOp> {
        &self.0
    }

    /// Remaps the axes of the given tree
    ///
    /// The remapping is lazy; it is not evaluated until the tree is imported
    /// into a `Context`.
    pub fn remap_xyz(&self, x: Tree, y: Tree, z: Tree) -> Tree {
        Self(Arc::new(TreeOp::RemapAxes {
            target: self.0.clone(),
            x: x.0,
            y: y.0,
            z: z.0,
        }))
    }
}

/// See [`Context`](crate::Context) for documentation of these functions
#[allow(missing_docs)]
impl Tree {
    pub fn x() -> Self {
        Tree(Arc::new(TreeOp::Input(Var::X)))
    }
    pub fn y() -> Self {
        Tree(Arc::new(TreeOp::Input(Var::Y)))
    }
    pub fn z() -> Self {
        Tree(Arc::new(TreeOp::Input(Var::Z)))
    }
    pub fn constant(f: f64) -> Self {
        Tree(Arc::new(TreeOp::Const(f)))
    }
    fn op_unary(a: Tree, op: UnaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Unary(op, a.0)))
    }
    fn op_binary(a: Tree, b: Tree, op: BinaryOpcode) -> Self {
        Tree(Arc::new(TreeOp::Binary(op, a.0, b.0)))
    }
    pub fn square(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Square)
    }
    pub fn floor(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Floor)
    }
    pub fn ceil(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Ceil)
    }
    pub fn round(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Round)
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
    pub fn compare<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Compare)
    }
    pub fn modulo<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Mod)
    }
    pub fn and<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::And)
    }
    pub fn or<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Or)
    }
    pub fn atan2<T: Into<Tree>>(&self, other: T) -> Self {
        Self::op_binary(self.clone(), other.into(), BinaryOpcode::Atan)
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
    pub fn not(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Not)
    }
    pub fn abs(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Abs)
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
                let mut next = self.clone().$base_fn(other.into());
                std::mem::swap(self, &mut next);
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
        // Remapping X
        let s = Tree::x() + 1.0;

        let v = s.remap_xyz(Tree::y(), Tree::z(), Tree::x());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::z(), Tree::x(), Tree::y());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 0.0, 1.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::x(), Tree::y(), Tree::z());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 2.0);

        // Remapping Y
        let s = Tree::y() + 1.0;

        let v = s.remap_xyz(Tree::y(), Tree::z(), Tree::x());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 0.0, 1.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::z(), Tree::x(), Tree::y());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::x(), Tree::y(), Tree::z());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 2.0);

        // Remapping Z
        let s = Tree::z() + 1.0;

        let v = s.remap_xyz(Tree::y(), Tree::z(), Tree::x());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::z(), Tree::x(), Tree::y());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 2.0);

        let v = s.remap_xyz(Tree::x(), Tree::y(), Tree::z());
        let mut ctx = Context::new();
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 0.0, 1.0).unwrap(), 2.0);

        // Test remapping to a constant
        let s = Tree::x() + 1.0;
        let one = Tree::constant(3.0);
        let v = s.remap_xyz(one, Tree::y(), Tree::z());
        let v_ = ctx.import(&v);
        assert_eq!(ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap(), 4.0);
    }

    #[test]
    fn deep_recursion_drop() {
        let mut x = Tree::x();
        for _ in 0..1_000_000 {
            x += 1.0;
        }
        drop(x);
        // we should not panic here!
    }

    #[test]
    fn deep_recursion_import() {
        let mut x = Tree::x();
        for _ in 0..1_000_000 {
            x += 1.0;
        }
        let mut ctx = Context::new();
        ctx.import(&x);
        // we should not panic here!
    }

    #[test]
    fn tree_remap_multi() {
        let mut ctx = Context::new();

        let out = Tree::x() + Tree::y() + Tree::z();
        let out =
            out.remap_xyz(Tree::x() * 2.0, Tree::y() * 3.0, Tree::z() * 5.0);

        let v_ = ctx.import(&out);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 1.0, 1.0).unwrap(), 10.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 1.0, 1.0).unwrap(), 12.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 2.0, 1.0).unwrap(), 15.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 2.0, 2.0).unwrap(), 20.0);

        let out = out.remap_xyz(Tree::y(), Tree::z(), Tree::x());
        let v_ = ctx.import(&out);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 1.0, 1.0).unwrap(), 10.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 1.0, 1.0).unwrap(), 15.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 2.0, 1.0).unwrap(), 17.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 2.0, 2.0).unwrap(), 20.0);
    }

    #[test]
    fn tree_import_cache() {
        let mut x = Tree::x();
        for _ in 0..100_000 {
            x += 1.0;
        }
        let mut ctx = Context::new();
        let start = std::time::Instant::now();
        ctx.import(&x);
        let small = start.elapsed();

        // Build a new tree with 4 copies of the original
        let x = x.clone() * x.clone() * x.clone() * x;
        let mut ctx = Context::new();
        let start = std::time::Instant::now();
        ctx.import(&x);
        let large = start.elapsed();

        assert!(
            large.as_millis() < small.as_millis() * 2,
            "tree import cache failed: {large:?} is much larger than {small:?}"
        );
    }

    #[test]
    fn tree_import_nocache() {
        let mut x = Tree::x();
        for _ in 0..100_000 {
            x += 1.0;
        }
        let mut ctx = Context::new();
        let start = std::time::Instant::now();
        ctx.import(&x);
        let small = start.elapsed();

        // Build a new tree with 4 remapped versions of the original
        let x = x.remap_xyz(Tree::y(), Tree::z(), Tree::x())
            * x.remap_xyz(Tree::z(), Tree::x(), Tree::y())
            * x.remap_xyz(Tree::y(), Tree::x(), Tree::z())
            * x;
        let mut ctx = Context::new();
        let start = std::time::Instant::now();
        ctx.import(&x);
        let large = start.elapsed();

        assert!(
            large.as_millis() > small.as_millis() * 2,
            "tree import cache failed:
             {large:?} is not much larger than {small:?}"
        );
    }

    #[test]
    fn tree_from_int() {
        let a = Tree::from(3);
        let b = a * 5;

        let mut ctx = Context::new();
        let root = ctx.import(&b);
        assert_eq!(ctx.get_const(root).unwrap(), 15.0);
    }
}
