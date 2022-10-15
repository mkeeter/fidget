use crate::tape::Tape;

/// Represents a point in space with associated partial derivatives.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Grad {
    v: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

impl Grad {
    pub fn abs(self) -> Self {
        if self.v < 0.0 {
            Grad {
                v: self.v,
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
        let v = 1.0 / self.v;
        let v2 = v * v;
        Grad {
            v,
            dx: -self.dx / v2,
            dy: -self.dy / v2,
            dz: -self.dz / v2,
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
        let d = rhs.v * rhs.v;
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

pub trait GradSliceFuncT {
    type Evaluator: GradSliceEvalT;

    fn from_tape(tape: Tape) -> Self;
    fn get_evaluator(&self) -> Self::Evaluator;
}

pub trait GradSliceEvalT {
    fn eval_g(&mut self, x: &[f32], y: &[f32], z: &[f32], out: &mut [Grad]);
}

/// Function handle for gradient slice evaluation
///
/// This trait represents a `struct` that _owns_ a function, but does not have
/// the equipment to evaluate it (e.g. scratch memory).  It is used to produce
/// one or more `GradSliceEval` objects, which actually do evaluation.
pub struct GradSliceFunc<F> {
    tape: Tape,
    func: F,
}

impl<F: GradSliceFuncT> GradSliceFunc<F> {
    pub fn from_tape(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            func: F::from_tape(tape),
        }
    }
    pub fn get_evaluator(
        &self,
    ) -> GradSliceEval<<F as GradSliceFuncT>::Evaluator> {
        GradSliceEval {
            tape: self.tape.clone(),
            eval: self.func.get_evaluator(),
        }
    }
}

pub struct GradSliceEval<E> {
    #[allow(dead_code)]
    pub(crate) tape: Tape,
    pub(crate) eval: E,
}

impl<E: GradSliceEvalT> GradSliceEval<E> {
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
        let mut out = [Grad::default()];
        self.eval_g(&[x], &[y], &[z], &mut out);
        out[0]
    }
}
