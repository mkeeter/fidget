use crate::{eval::EvalFamily, tape::Tape};

/// Represents a point in space with associated partial derivatives.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Grad {
    v: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

impl Grad {
    pub fn new(v: f32, dx: f32, dy: f32, dz: f32) -> Self {
        Self { v, dx, dy, dz }
    }

    /// Returns a normalized RGB color, or `None` if the gradient is 0
    pub fn to_rgb(&self) -> Option<[u8; 3]> {
        let s = (self.dx.powi(2) + self.dy.powi(2) + self.dz.powi(2)).sqrt();
        if s != 0.0 {
            let scale = u8::MAX as f32 / s;
            Some([
                (self.dx.abs() * scale) as u8,
                (self.dy.abs() * scale) as u8,
                (self.dz.abs() * scale) as u8,
            ])
        } else {
            None
        }
    }

    pub fn abs(self) -> Self {
        if self.v < 0.0 {
            Grad {
                v: -self.v,
                dx: -self.dx,
                dy: -self.dy,
                dz: -self.dz,
            }
        } else {
            self
        }
    }
    pub fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Grad {
            v,
            dx: self.dx / (2.0 * v),
            dy: self.dy / (2.0 * v),
            dz: self.dz / (2.0 * v),
        }
    }
    pub fn recip(self) -> Self {
        let v2 = -self.v.powi(2);
        Grad {
            v: 1.0 / self.v,
            dx: self.dx / v2,
            dy: self.dy / v2,
            dz: self.dz / v2,
        }
    }
    pub fn min(self, rhs: Self) -> Self {
        if self.v < rhs.v {
            self
        } else {
            rhs
        }
    }
    pub fn max(self, rhs: Self) -> Self {
        if self.v > rhs.v {
            self
        } else {
            rhs
        }
    }
}

impl From<f32> for Grad {
    fn from(v: f32) -> Self {
        Grad {
            v,
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }
}

impl std::ops::Add<Grad> for Grad {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Grad {
            v: self.v + rhs.v,
            dx: self.dx + rhs.dx,
            dy: self.dy + rhs.dy,
            dz: self.dz + rhs.dz,
        }
    }
}

impl std::ops::Mul<Grad> for Grad {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            dx: self.v * rhs.dx + rhs.v * self.dx,
            dy: self.v * rhs.dy + rhs.v * self.dy,
            dz: self.v * rhs.dz + rhs.v * self.dz,
        }
    }
}

impl std::ops::Div<Grad> for Grad {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.v.powi(2);
        Self {
            v: self.v / rhs.v,
            dx: (rhs.v * self.dx - self.v * rhs.dx) / d,
            dy: (rhs.v * self.dy - self.v * rhs.dy) / d,
            dz: (rhs.v * self.dz - self.v * rhs.dz) / d,
        }
    }
}

impl std::ops::Sub<Grad> for Grad {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
            dz: self.dz - rhs.dz,
        }
    }
}

impl std::ops::Neg for Grad {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            dx: -self.dx,
            dy: -self.dy,
            dz: -self.dz,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait GradEvalT: From<Tape> {
    type Family: EvalFamily;
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> Grad;
    fn eval_g(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [Grad]) {
        let len = [x.len(), y.len(), z.len(), out.len()]
            .into_iter()
            .min()
            .unwrap();
        for i in 0..len {
            out[i] = self.eval_f(x[i], y[i], z[i]);
        }
    }
}

pub struct GradEval<E> {
    #[allow(dead_code)]
    tape: Tape,
    eval: E,
}

impl<E: GradEvalT> From<Tape> for GradEval<E> {
    fn from(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            eval: E::from(tape),
        }
    }
}

impl<E: GradEvalT> GradEval<E> {
    pub fn eval_g(
        &mut self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        out: &mut [Grad],
    ) {
        self.eval.eval_g(x, y, z, out)
    }
    pub fn eval_f(&mut self, x: f32, y: f32, z: f32) -> Grad {
        self.eval.eval_f(x, y, z)
    }
}

#[cfg(feature = "eval-tests")]
pub mod eval_tests {
    use super::*;
    use crate::context::Context;

    pub fn test_g_x<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let tape = ctx.get_tape(x, I::Family::REG_LIMIT);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(0.0, 0.0, 0.0), Grad::new(0.0, 1.0, 0.0, 0.0));
    }

    pub fn test_g_square<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.square(x).unwrap();
        let tape = ctx.get_tape(s, I::Family::REG_LIMIT);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(0.0, 0.0, 0.0), Grad::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(1.0, 2.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(2.0, 0.0, 0.0), Grad::new(4.0, 4.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(3.0, 0.0, 0.0), Grad::new(9.0, 6.0, 0.0, 0.0));
    }

    pub fn test_g_sqrt<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.sqrt(x).unwrap();
        let tape = ctx.get_tape(s, I::Family::REG_LIMIT);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(1.0, 0.5, 0.0, 0.0));
        assert_eq!(eval.eval_f(4.0, 0.0, 0.0), Grad::new(2.0, 0.25, 0.0, 0.0));
    }

    pub fn test_g_mul<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let s = ctx.mul(x, y).unwrap();
        let tape = ctx.get_tape(s, I::Family::REG_LIMIT);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(eval.eval_f(0.0, 1.0, 0.0), Grad::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(4.0, 1.0, 0.0), Grad::new(4.0, 1.0, 4.0, 0.0));
        assert_eq!(eval.eval_f(4.0, 2.0, 0.0), Grad::new(8.0, 2.0, 4.0, 0.0));
    }

    pub fn test_g_recip<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let s = ctx.recip(x).unwrap();
        let tape = ctx.get_tape(s, I::Family::REG_LIMIT);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(1.0, -1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(2.0, 0.0, 0.0), Grad::new(0.5, -0.25, 0.0, 0.0));
    }

    pub fn test_g_circle<I: GradEvalT>() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let sqrt = ctx.sqrt(sum).unwrap();
        let half = ctx.constant(0.5);
        let sub = ctx.sub(sqrt, half).unwrap();
        let tape = ctx.get_tape(sub, u8::MAX);

        let mut eval = GradEval::<I>::from(tape);
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(0.5, 1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(0.0, 1.0, 0.0), Grad::new(0.5, 0.0, 1.0, 0.0));
        assert_eq!(eval.eval_f(2.0, 0.0, 0.0), Grad::new(1.5, 1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(0.0, 2.0, 0.0), Grad::new(1.5, 0.0, 1.0, 0.0));
    }

    #[macro_export]
    macro_rules! grad_test {
        ($i:ident, $t:ty) => {
            #[test]
            fn $i() {
                $crate::eval::grad::eval_tests::$i::<$t>()
            }
        };
    }

    #[macro_export]
    macro_rules! grad_tests {
        ($t:ty) => {
            $crate::grad_test!(test_g_circle, $t);
            $crate::grad_test!(test_g_x, $t);
            $crate::grad_test!(test_g_square, $t);
            $crate::grad_test!(test_g_sqrt, $t);
            $crate::grad_test!(test_g_mul, $t);
            $crate::grad_test!(test_g_recip, $t);
        };
    }
}
