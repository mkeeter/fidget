use crate::{
    context::{Context, IntoNode, Node},
    eval::{Family, Tape},
    Error,
};
use std::{cell::RefCell, rc::Rc};

/// Shareable context used in a [`BoundNode`]
///
/// This is only available for unit testing, because it is less efficient than
/// manually managing a [`Context`].
#[derive(Clone)]
pub struct BoundContext(Rc<RefCell<Context>>);

impl std::ops::Deref for BoundContext {
    type Target = Rc<RefCell<Context>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for BoundContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl BoundContext {
    /// Creates a new, empty `BoundContext`
    pub fn new() -> Self {
        BoundContext(Rc::new(RefCell::new(Context::new())))
    }
    /// Returns `(x, y, z)` axes from this context
    pub fn axes(&self) -> (BoundNode, BoundNode, BoundNode) {
        let x = self.borrow_mut().x();
        let y = self.borrow_mut().y();
        let z = self.borrow_mut().z();
        let x = BoundNode {
            node: x,
            ctx: self.clone(),
        };
        let y = BoundNode {
            node: y,
            ctx: self.clone(),
        };
        let z = BoundNode {
            node: z,
            ctx: self.clone(),
        };
        (x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Represents a node bound to a specific context
///
/// This allows us to write inline math expressions, for ease of testing.
/// However, it's less efficient: each `BoundNode` doubles in size, and every
/// operation encurs a `RefCell` dynamic borrow.
#[derive(Clone)]
pub struct BoundNode {
    node: Node,
    ctx: BoundContext,
}

impl IntoNode for BoundNode {
    fn into_node(self, ctx: &mut Context) -> Result<Node, Error> {
        assert_eq!(self.ctx.as_ptr(), ctx as *mut Context);
        Ok(self.node)
    }
}

impl BoundNode {
    fn op_bin<A: IntoNode>(
        self,
        other: A,
        f: fn(&mut Context, Node, Node) -> Result<Node, Error>,
    ) -> Self {
        let node = {
            let mut ctx = self.ctx.borrow_mut();
            let other = other.into_node(&mut ctx).unwrap();
            f(&mut ctx, self.node, other).unwrap()
        };
        BoundNode {
            node,
            ctx: self.ctx,
        }
    }

    fn op_unary(
        self,
        f: fn(&mut Context, Node) -> Result<Node, Error>,
    ) -> Self {
        let node = {
            let mut ctx = self.ctx.borrow_mut();
            f(&mut ctx, self.node).unwrap()
        };
        BoundNode {
            node,
            ctx: self.ctx,
        }
    }

    /// Builds a `min` operation
    ///
    /// # Panics
    /// If the nodes are invalid or are not from the same [`Context`]
    pub fn min<A: IntoNode>(self, other: A) -> Self {
        self.op_bin(other, Context::min)
    }

    /// Builds a `max` operation
    ///
    /// # Panics
    /// If the nodes are invalid or are not from the same [`Context`]
    pub fn max<A: IntoNode>(self, other: A) -> Self {
        self.op_bin(other, Context::max)
    }

    /// Builds a `square` operation
    pub fn square(self) -> Self {
        self.op_unary(Context::square)
    }

    /// Builds a square root operation
    pub fn sqrt(self) -> Self {
        self.op_unary(Context::sqrt)
    }

    /// Converts this node into a tape, using its internal context
    pub fn get_tape<E: Family>(&self) -> Result<Tape<E>, Error> {
        self.ctx.borrow().get_tape::<E>(self.node)
    }
}

macro_rules! impl_binary {
    ($op:ident, $fn_name:ident, $ctx_name:ident) => {
        impl<A: IntoNode> std::ops::$op<A> for BoundNode {
            type Output = Self;

            fn $fn_name(self, other: A) -> Self {
                self.op_bin(other, Context::$ctx_name)
            }
        }
        impl std::ops::$op<BoundNode> for f32 {
            type Output = BoundNode;

            fn $fn_name(self, other: BoundNode) -> Self::Output {
                let lhs = BoundNode {
                    node: self.into_node(&mut other.ctx.borrow_mut()).unwrap(),
                    ctx: other.ctx.clone(),
                };
                lhs.op_bin(other, Context::$ctx_name)
            }
        }
    };
    ($op:ident, $fn_name:ident) => {
        impl_binary!($op, $fn_name, $fn_name);
    };
}

impl_binary!(Add, add);
impl_binary!(Sub, sub);
impl_binary!(Mul, mul);
impl_binary!(Div, div);

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bound_node() {
        let (x, y, z) = BoundContext::new().axes();
        assert_eq!(x.node.0, 0);
        assert_eq!(y.node.0, 1);
        assert_eq!(z.node.0, 2);
        let n = x + y + z + 1.0;
        assert_eq!(n.node.0, 6);
    }
}
