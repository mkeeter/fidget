//! Context-free math trees
use super::op::{BinaryOpcode, UnaryOpcode};
use crate::{Error, var::Var};
use std::{cmp::Ordering, sync::Arc};

/// Opcode type for trees
///
/// This is equivalent to [`Op`](crate::context::Op), but also includes the
/// [`RemapAxes`](TreeOp::RemapAxes) and [`TreeOp::RemapAffine`] operations for
/// lazy remapping.
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
    ///
    /// If the transform is affine, then `RemapAffine` should be preferred,
    /// because it flattens sequences of affine transformations.
    RemapAxes {
        target: Arc<TreeOp>,
        x: Arc<TreeOp>,
        y: Arc<TreeOp>,
        z: Arc<TreeOp>,
    },
    /// Lazy affine transforms
    ///
    /// When imported into a `Context`, the `x/y/z` clauses within `target` will
    /// be transformed with the provided affine matrix.
    RemapAffine {
        target: Arc<TreeOp>,
        mat: nalgebra::Affine3<f64>,
    },
}

impl Drop for TreeOp {
    fn drop(&mut self) {
        // Early exit for TreeOps which have limited recursion
        if self.eligible_for_fast_drop() {
            return;
        }

        let mut todo = vec![std::mem::replace(self, TreeOp::Const(0.0))];
        let empty = Arc::new(TreeOp::Const(0.0));
        while let Some(mut t) = todo.pop() {
            for t in t.iter_children_mut() {
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
    /// eligible for fast dropping if all of its children are non-recursive.
    fn eligible_for_fast_drop(&self) -> bool {
        self.iter_children().all(|c| c.does_not_recurse())
    }

    /// Returns `true` if the given child does not recurse
    fn does_not_recurse(&self) -> bool {
        matches!(self, TreeOp::Const(..) | TreeOp::Input(..))
    }

    fn iter_children(&self) -> impl Iterator<Item = &Arc<TreeOp>> {
        match self {
            TreeOp::Const(..) | TreeOp::Input(..) => [None, None, None, None],
            TreeOp::Unary(_op, arg) => [Some(arg), None, None, None],
            TreeOp::Binary(_op, lhs, rhs) => [Some(lhs), Some(rhs), None, None],
            TreeOp::RemapAxes { target, x, y, z } => {
                [Some(target), Some(x), Some(y), Some(z)]
            }
            TreeOp::RemapAffine { target, .. } => {
                [Some(target), None, None, None]
            }
        }
        .into_iter()
        .flatten()
    }

    fn iter_children_mut(&mut self) -> impl Iterator<Item = &mut Arc<TreeOp>> {
        match self {
            TreeOp::Const(..) | TreeOp::Input(..) => [None, None, None, None],
            TreeOp::Unary(_op, arg) => [Some(arg), None, None, None],
            TreeOp::Binary(_op, lhs, rhs) => [Some(lhs), Some(rhs), None, None],
            TreeOp::RemapAxes { target, x, y, z } => {
                [Some(target), Some(x), Some(y), Some(z)]
            }
            TreeOp::RemapAffine { target, .. } => {
                [Some(target), None, None, None]
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

impl From<TreeOp> for Tree {
    fn from(t: TreeOp) -> Tree {
        Tree(Arc::new(t))
    }
}

/// Owned handle for a standalone math tree
#[derive(Clone, Debug, facet::Facet)]
pub struct Tree(#[facet(opaque)] Arc<TreeOp>);

impl std::ops::Deref for Tree {
    type Target = TreeOp;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        // Heap recursion using a `Vec`, to avoid blowing up the stack
        let mut todo = vec![(&self.0, &other.0)];
        while let Some((a, b)) = todo.pop() {
            // Pointer equality lets us short-circuit deep checks
            if Arc::as_ptr(a) == Arc::as_ptr(b) {
                continue;
            }
            // Otherwise, we check opcodes then recurse
            match (a.as_ref(), b.as_ref()) {
                (TreeOp::Input(a), TreeOp::Input(b)) => {
                    if *a != *b {
                        return false;
                    }
                }
                (TreeOp::Const(a), TreeOp::Const(b)) => {
                    if *a != *b {
                        return false;
                    }
                }
                (TreeOp::Unary(op_a, arg_a), TreeOp::Unary(op_b, arg_b)) => {
                    if *op_a != *op_b {
                        return false;
                    }
                    todo.push((arg_a, arg_b));
                }
                (
                    TreeOp::Binary(op_a, lhs_a, rhs_a),
                    TreeOp::Binary(op_b, lhs_b, rhs_b),
                ) => {
                    if *op_a != *op_b {
                        return false;
                    }
                    todo.push((lhs_a, lhs_b));
                    todo.push((rhs_a, rhs_b));
                }
                (
                    TreeOp::RemapAxes {
                        target: t_a,
                        x: x_a,
                        y: y_a,
                        z: z_a,
                    },
                    TreeOp::RemapAxes {
                        target: t_b,
                        x: x_b,
                        y: y_b,
                        z: z_b,
                    },
                ) => {
                    todo.push((t_a, t_b));
                    todo.push((x_a, x_b));
                    todo.push((y_a, y_b));
                    todo.push((z_a, z_b));
                }
                (
                    TreeOp::RemapAffine {
                        target: t_a,
                        mat: mat_a,
                    },
                    TreeOp::RemapAffine {
                        target: t_b,
                        mat: mat_b,
                    },
                ) => {
                    if *mat_a != *mat_b {
                        return false;
                    }
                    todo.push((t_a, t_b));
                }
                _ => return false,
            }
        }
        true
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

    /// Shallow (pointer) equality check
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.as_ptr(), other.as_ptr())
    }

    /// Borrow the inner `Arc<TreeOp>`
    pub(crate) fn arc(&self) -> &Arc<TreeOp> {
        &self.0
    }

    /// Remaps the axes of the given tree
    ///
    /// If the mapping is affine, then [`remap_affine`](Self::remap_affine)
    /// should be preferred.
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

    /// Performs an affine remapping of the given tree
    ///
    /// The remapping is lazy; it is not evaluated until the tree is imported
    /// into a `Context`.
    pub fn remap_affine(&self, mat: nalgebra::Affine3<f64>) -> Tree {
        // Flatten affine trees
        let out = match &*self.0 {
            TreeOp::RemapAffine { target, mat: next } => TreeOp::RemapAffine {
                target: target.clone(),
                mat: next * mat,
            },
            _ => TreeOp::RemapAffine {
                target: self.0.clone(),
                mat,
            },
        };
        Self(out.into())
    }

    /// Returns the inner [`Var`] if this is an input tree, or `None`
    pub fn var(&self) -> Option<Var> {
        if let TreeOp::Input(v) = &*self.0 {
            Some(*v)
        } else {
            None
        }
    }

    /// Performs symbolic differentiation with respect to the given variable
    pub fn deriv(&self, v: Var) -> Tree {
        let mut ctx = crate::Context::new();
        let node = ctx.import(self);
        ctx.deriv(node, v).and_then(|d| ctx.export(d)).unwrap()
    }

    /// Raises this tree to the power of an integer using exponentiation by squaring
    pub fn pow(&self, mut n: i64) -> Self {
        // TODO should this also be in `Context`?
        let mut x = match n.cmp(&0) {
            Ordering::Less => {
                n = -n;
                self.recip()
            }
            Ordering::Equal => {
                return Tree::from(1.0);
            }
            Ordering::Greater => self.clone(),
        };
        let mut y: Option<Tree> = None;
        while n > 1 {
            if n % 2 == 1 {
                y = match y {
                    Some(y) => Some(x.clone() * y),
                    None => Some(x.clone()),
                };
                n -= 1;
            }
            x = x.square();
            n /= 2;
        }
        if let Some(y) = y {
            x *= y;
        }
        x
    }
}

impl TryFrom<Tree> for Var {
    type Error = Error;
    fn try_from(t: Tree) -> Result<Var, Error> {
        t.var().ok_or(Error::NotAVar)
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
    pub fn recip(&self) -> Self {
        Self::op_unary(self.clone(), UnaryOpcode::Recip)
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

impl std::ops::Neg for Tree {
    type Output = Tree;
    fn neg(self) -> Self::Output {
        Tree::op_unary(self, UnaryOpcode::Neg)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Context;

    #[test]
    fn tree_x() {
        let x1 = Tree::x();
        let x2 = Tree::x();
        assert!(!x1.ptr_eq(&x2)); // shallow equality
        assert_eq!(x1, x2); // deep equality

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
    fn test_remap_affine() {
        let s = Tree::x();
        // Two rotations by 45° -> 90°
        let t = nalgebra::convert(nalgebra::Rotation3::<f64>::from_axis_angle(
            &nalgebra::Vector3::<f64>::z_axis(),
            -std::f64::consts::FRAC_PI_4,
        ));
        let s = s.remap_affine(t);
        let s = s.remap_affine(t);

        let TreeOp::RemapAffine { target, .. } = &*s else {
            panic!("invalid shape");
        };
        assert!(matches!(&**target, TreeOp::Input(Var::X)));

        let mut ctx = Context::new();
        let v_ = ctx.import(&s);

        assert!((ctx.eval_xyz(v_, 0.0, 1.0, 0.0).unwrap() - 1.0).abs() < 1e-6);
        assert!(
            (ctx.eval_xyz(v_, 0.0, -2.0, 0.0).unwrap() - -2.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_remap_order() {
        let translate = nalgebra::convert(nalgebra::Translation3::<f64>::new(
            3.0, 0.0, 0.0,
        ));
        let scale =
            nalgebra::convert(nalgebra::Scale3::<f64>::new(0.5, 0.5, 0.5));

        let s = Tree::x();
        let s = s.remap_affine(translate);
        let s = s.remap_affine(scale);

        // Confirm that we didn't stack up RemapAffine nodes
        let TreeOp::RemapAffine { target, .. } = &*s else {
            panic!("invalid shape");
        };
        assert!(matches!(&**target, TreeOp::Input(Var::X)));

        // Basic evaluation testing
        let mut ctx = Context::new();
        let v_ = ctx.import(&s);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 3.5);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 0.0, 0.0).unwrap(), 4.0);

        // Do the same thing but testing collapsing in `Context::import`
        let manual = TreeOp::RemapAffine {
            target: Arc::new(TreeOp::RemapAffine {
                target: TreeOp::Input(Var::X).into(),
                mat: scale,
            }),
            mat: translate,
        }
        .into();
        let mut ctx = Context::new();
        let v_ = ctx.import(&manual);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 3.5);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 0.0, 0.0).unwrap(), 4.0);

        // Swap the order and make sure it still works
        let s = Tree::x();
        let s = s.remap_affine(scale);
        let s = s.remap_affine(translate);

        let mut ctx = Context::new();
        let v_ = ctx.import(&s);
        assert_eq!(ctx.eval_xyz(v_, 1.0, 0.0, 0.0).unwrap(), 2.0);
        assert_eq!(ctx.eval_xyz(v_, 2.0, 0.0, 0.0).unwrap(), 2.5);
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
    fn deep_recursion_eq() {
        let mut x1 = Tree::x();
        for _ in 0..1_000_000 {
            x1 += 1.0;
        }
        let mut x2 = Tree::x();
        for _ in 0..1_000_000 {
            x2 += 1.0;
        }
        assert_eq!(x1, x2);
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

    #[test]
    fn tree_deriv() {
        // dx/dx = 1
        let x = Tree::x();
        let vx = x.var().unwrap();
        let d = x.deriv(vx);
        let TreeOp::Const(v) = *d else {
            panic!("invalid deriv {d:?}")
        };
        assert_eq!(v, 1.0);

        // dx/dv = 0
        let d = x.deriv(Var::new());
        let TreeOp::Const(v) = *d else {
            panic!("invalid deriv {d:?}")
        };
        assert_eq!(v, 0.0);
    }

    #[test]
    fn tree_pow() {
        let a = Tree::from(3);
        let b = a.pow(3);
        let c = a.pow(-3);
        let d = a.pow(0);

        let mut ctx = Context::new();
        let root = ctx.import(&b);
        assert_eq!(ctx.get_const(root).unwrap(), 27.0);
        ctx.clear();
        let root = ctx.import(&c);
        assert_eq!(ctx.get_const(root).unwrap(), 1.0 / 27.0);
        ctx.clear();
        let root = ctx.import(&d);
        assert_eq!(ctx.get_const(root).unwrap(), 1.0);
    }

    #[test]
    fn tree_poke() {
        use facet::Facet;
        #[derive(facet::Facet)]
        struct Transform {
            tree: Tree,
            x: f64,
        }

        let builder = facet::Partial::alloc_shape(Transform::SHAPE)
            .unwrap()
            .set_field("tree", Tree::x() + 2.0 * Tree::y())
            .unwrap()
            .set_field("x", 1.0)
            .unwrap();
        let t: Transform = builder.build().unwrap().materialize().unwrap();
        assert_eq!(t.x, 1.0);
        let mut ctx = Context::new();
        let node = ctx.import(&t.tree);
        assert_eq!(ctx.eval_xyz(node, 1.0, 2.0, 3.0).unwrap(), 5.0);
    }
}
