//! Infrastructure and algorithms for complex closed-form implicit surfaces.
//!
//! ```
//! use jitfive::context::Context;
//! let mut ctx = Context::new();
//! let x = ctx.x();
//! let y = ctx.y();
//! let x_squared = ctx.mul(x, x).unwrap();
//! let y_squared = ctx.mul(y, y).unwrap();
//! let radius = ctx.add(x_squared, y_squared).unwrap();
//! let one = ctx.constant(1.0);
//! let circle = ctx.sub(radius, one).unwrap();
//!
//! use jitfive::asm::dynasm::REGISTER_LIMIT;
//! let tape = ctx.get_tape(circle, REGISTER_LIMIT);
//! ```
pub mod asm;
pub mod context;
pub mod render;
pub mod tape;

mod error;

pub use error::Error;

#[cfg(test)]
mod tests {
    use crate::context::*;

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
        let b = ctx.constant(-1.0);
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
