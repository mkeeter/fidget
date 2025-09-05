/// A point in space with associated partial derivatives.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Grad {
    /// Value of the distance field at this point
    pub v: f32,
    /// Partial derivative with respect to `x`
    pub dx: f32,
    /// Partial derivative with respect to `y`
    pub dy: f32,
    /// Partial derivative with respect to `z`
    pub dz: f32,
}

impl std::fmt::Display for Grad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.v, self.dx, self.dy, self.dz)
    }
}

impl Grad {
    /// Constructs a new gradient
    #[inline]
    pub fn new(v: f32, dx: f32, dy: f32, dz: f32) -> Self {
        Self { v, dx, dy, dz }
    }

    /// Looks up a gradient by index (0 = x, 1 = y, 2 = z)
    ///
    /// # Panics
    /// If the index is not in the 0-2 range
    #[inline]
    pub fn d(&self, i: usize) -> f32 {
        match i {
            0 => self.dx,
            1 => self.dy,
            2 => self.dz,
            _ => panic!("invalid index {i}"),
        }
    }

    /// Absolute value
    #[inline]
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

    /// Square root
    #[inline]
    pub fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Grad {
            v,
            dx: self.dx / (2.0 * v),
            dy: self.dy / (2.0 * v),
            dz: self.dz / (2.0 * v),
        }
    }

    /// Sine
    #[inline]
    pub fn sin(self) -> Self {
        let c = self.v.cos();
        Grad {
            v: self.v.sin(),
            dx: self.dx * c,
            dy: self.dy * c,
            dz: self.dz * c,
        }
    }
    /// Cosine
    #[inline]
    pub fn cos(self) -> Self {
        let s = -self.v.sin();
        Grad {
            v: self.v.cos(),
            dx: self.dx * s,
            dy: self.dy * s,
            dz: self.dz * s,
        }
    }
    /// Tangent
    #[inline]
    pub fn tan(self) -> Self {
        let c = self.v.cos().powi(2);
        Grad {
            v: self.v.tan(),
            dx: self.dx / c,
            dy: self.dy / c,
            dz: self.dz / c,
        }
    }
    /// Arcsin
    #[inline]
    pub fn asin(self) -> Self {
        let r = (1.0 - self.v.powi(2)).sqrt();
        Grad {
            v: self.v.asin(),
            dx: self.dx / r,
            dy: self.dy / r,
            dz: self.dz / r,
        }
    }
    /// Arccos
    #[inline]
    pub fn acos(self) -> Self {
        let r = (1.0 - self.v.powi(2)).sqrt();
        Grad {
            v: self.v.acos(),
            dx: -self.dx / r,
            dy: -self.dy / r,
            dz: -self.dz / r,
        }
    }
    /// Arctangent
    #[inline]
    pub fn atan(self) -> Self {
        let r = self.v.powi(2) + 1.0;
        Grad {
            v: self.v.atan(),
            dx: self.dx / r,
            dy: self.dy / r,
            dz: self.dz / r,
        }
    }
    /// Exponential function
    #[inline]
    pub fn exp(self) -> Self {
        let v = self.v.exp();
        Grad {
            v,
            dx: v * self.dx,
            dy: v * self.dy,
            dz: v * self.dz,
        }
    }
    /// Natural log
    #[inline]
    pub fn ln(self) -> Self {
        Grad {
            v: self.v.ln(),
            dx: self.dx / self.v,
            dy: self.dy / self.v,
            dz: self.dz / self.v,
        }
    }

    /// Reciprocal
    #[inline]
    pub fn recip(self) -> Self {
        let v2 = -self.v.powi(2);
        Grad {
            v: 1.0 / self.v,
            dx: self.dx / v2,
            dy: self.dy / v2,
            dz: self.dz / v2,
        }
    }

    /// Minimum of two values
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        if self.v < rhs.v { self } else { rhs }
    }

    /// Maximum of two values
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        if self.v > rhs.v { self } else { rhs }
    }

    /// Least non-negative remainder
    #[inline]
    pub fn rem_euclid(&self, rhs: Grad) -> Self {
        let e = self.v.div_euclid(rhs.v);
        Grad {
            v: self.v.rem_euclid(rhs.v),
            dx: self.dx - rhs.dx * e,
            dy: self.dy - rhs.dy * e,
            dz: self.dz - rhs.dz * e,
        }
    }

    /// Snap to the largest less-than-or-equal value
    #[inline]
    pub fn floor(&self) -> Self {
        Grad {
            v: self.v.floor(),
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }

    /// Snap to the smallest greater-than-or-equal value
    #[inline]
    pub fn ceil(&self) -> Self {
        Grad {
            v: self.v.ceil(),
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }

    /// Rounds to the nearest integer
    #[inline]
    pub fn round(&self) -> Self {
        Grad {
            v: self.v.round(),
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }

    /// Four-quadrant arctangent
    #[inline]
    pub fn atan2(self, x: Self) -> Self {
        let y = self;

        let d = x.v.powi(2) + y.v.powi(2);
        Grad {
            v: y.v.atan2(x.v),
            dx: (x.v * y.dx - y.v * x.dx) / d,
            dy: (x.v * y.dy - y.v * x.dy) / d,
            dz: (x.v * y.dz - y.v * x.dz) / d,
        }
    }

    /// Checks that the two values are roughly equal, panicking otherwise
    #[cfg(test)]
    pub(crate) fn compare_eq(&self, other: Self) {
        let d = (self.v - other.v)
            .abs()
            .max((self.dx - other.dx).abs())
            .max((self.dy - other.dy).abs())
            .max((self.dz - other.dz).abs());
        if d >= 1e-6 {
            panic!("lhs != rhs ({self:?} != {other:?})");
        }
    }
}

impl From<f32> for Grad {
    #[inline]
    fn from(v: f32) -> Self {
        Grad {
            v,
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }
}

impl From<Grad> for nalgebra::Vector4<f32> {
    #[inline]
    fn from(g: Grad) -> Self {
        nalgebra::Vector4::new(g.dx, g.dy, g.dz, g.v)
    }
}

impl std::ops::Add<Grad> for Grad {
    type Output = Self;

    #[inline]
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

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            dx: self.v * rhs.dx + rhs.v * self.dx,
            dy: self.v * rhs.dy + rhs.v * self.dy,
            dz: self.v * rhs.dz + rhs.v * self.dz,
        }
    }
}

impl std::ops::Mul<f32> for Grad {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            v: self.v * rhs,
            dx: self.dx * rhs,
            dy: self.dy * rhs,
            dz: self.dz * rhs,
        }
    }
}

impl std::ops::Div<Grad> for Grad {
    type Output = Self;

    #[inline]
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

    #[inline]
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

    #[inline]
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            dx: -self.dx,
            dy: -self.dy,
            dz: -self.dz,
        }
    }
}
