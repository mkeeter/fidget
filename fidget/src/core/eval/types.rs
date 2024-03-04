//! Custom types used during evaluation
use crate::vm::Choice;

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

impl Grad {
    /// Constructs a new gradient
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

    /// Absolute value
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
    pub fn ln(self) -> Self {
        let v = self.v.exp();
        Grad {
            v,
            dx: self.dx / v,
            dy: self.dy / v,
            dz: self.dz / v,
        }
    }

    /// Reciprocal
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
    pub fn min(self, rhs: Self) -> Self {
        if self.v < rhs.v {
            self
        } else {
            rhs
        }
    }

    /// Maximum of two values
    pub fn max(self, rhs: Self) -> Self {
        if self.v > rhs.v {
            self
        } else {
            rhs
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
    fn from(g: Grad) -> Self {
        nalgebra::Vector4::new(g.dx, g.dy, g.dz, g.v)
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

/// Stores a range, with conservative calculations to guarantee that it always
/// contains the actual value.
///
/// # Warning
/// This implementation does not set rounding modes, so it may not be _perfect_.
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Interval {
    lower: f32,
    upper: f32,
}

impl std::fmt::Debug for Interval {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        f.debug_tuple("")
            .field(&self.lower)
            .field(&self.upper)
            .finish()
    }
}

impl Interval {
    /// Builds a new interval
    ///
    /// There are two kinds of valid interval:
    /// - `[lower, upper]` where `lower <= upper`
    /// - `[NaN, NaN]`
    ///
    /// # Panics
    /// Panics if the resulting interval would be invalid
    #[inline]
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            upper >= lower || (lower.is_nan() && upper.is_nan()),
            "invalid interval [{lower}, {upper}]"
        );
        Self { lower, upper }
    }
    /// Returns the lower bound of the interval
    #[inline]
    pub fn lower(&self) -> f32 {
        self.lower
    }
    /// Returns the upper bound of the interval
    #[inline]
    pub fn upper(&self) -> f32 {
        self.upper
    }
    /// Checks whether the given value is (strictly) contained in the interval
    #[inline]
    pub fn contains(&self, v: f32) -> bool {
        v >= self.lower && v <= self.upper
    }
    /// Returns `true` if either bound of the interval is `NaN`
    pub fn has_nan(&self) -> bool {
        self.lower.is_nan() || self.upper.is_nan()
    }
    /// Calculates the absolute value of the interval
    pub fn abs(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval::new(0.0, self.upper.max(-self.lower))
            } else {
                Interval::new(-self.upper, -self.lower)
            }
        } else {
            self
        }
    }
    /// Squares the interval
    ///
    /// Note that this has tighter bounds than multiplication, because we know
    /// that both sides of the multiplication are the same value.
    pub fn square(self) -> Self {
        if self.upper < 0.0 {
            Interval::new(self.upper.powi(2), self.lower.powi(2))
        } else if self.lower > 0.0 {
            Interval::new(self.lower.powi(2), self.upper.powi(2))
        } else if self.has_nan() {
            f32::NAN.into()
        } else {
            Interval::new(0.0, self.lower.abs().max(self.upper.abs()).powi(2))
        }
    }
    /// Computes the sine of the interval
    ///
    /// Right now, this always returns the maximum range of `[-1, 1]`
    pub fn sin(self) -> Self {
        // TODO: make this smarter
        Interval::new(-1.0, 1.0)
    }
    /// Computes the cosine of the interval
    ///
    /// Right now, this always returns the maximum range of `[-1, 1]`
    pub fn cos(self) -> Self {
        // TODO: make this smarter
        Interval::new(-1.0, 1.0)
    }
    /// Computes the tangent of the interval
    ///
    /// Returns the `NAN` interval if the result contains a undefined point
    pub fn tan(self) -> Self {
        let size = self.upper - self.lower;
        if size >= std::f32::consts::PI {
            f32::NAN.into()
        } else {
            let lower = self.lower.atan();
            let upper = self.upper.atan();
            if upper >= lower {
                Interval::new(lower, upper)
            } else {
                f32::NAN.into()
            }
        }
    }
    /// Computes the arcsine of the interval
    ///
    /// Returns the `NAN` interval if the input is invalid
    pub fn asin(self) -> Self {
        if self.lower < -1.0 || self.upper > 1.0 {
            f32::NAN.into()
        } else {
            Interval::new(self.lower.asin(), self.upper.asin())
        }
    }
    /// Computes the arccosine of the interval
    ///
    /// Returns the `NAN` interval if the input is invalid
    pub fn acos(self) -> Self {
        if self.lower < -1.0 || self.upper > 1.0 {
            f32::NAN.into()
        } else {
            Interval::new(self.upper.acos(), self.lower.acos())
        }
    }
    /// Computes the arctangent of the interval
    pub fn atan(self) -> Self {
        Interval::new(self.upper.atan(), self.lower.atan())
    }
    /// Computes the exponent function applied to the interval
    pub fn exp(self) -> Self {
        Interval::new(self.lower.exp(), self.upper.exp())
    }
    /// Computes the natural log of the input interval
    ///
    /// Returns the `NAN` interval if the input contains zero
    pub fn ln(self) -> Self {
        if self.lower <= 0.0 {
            f32::NAN.into()
        } else {
            Interval::new(self.upper.ln(), self.lower.ln())
        }
    }
    /// Calculates the square root of the interval
    ///
    /// If the entire interval is below 0, returns a `NAN` interval; otherwise,
    /// returns the valid (positive) interval.
    pub fn sqrt(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval::new(0.0, self.upper.sqrt())
            } else {
                f32::NAN.into()
            }
        } else {
            Interval::new(self.lower.sqrt(), self.upper.sqrt())
        }
    }
    /// Calculates the reciprocal of the interval
    ///
    /// If the interval includes 0, returns the `NAN` interval
    pub fn recip(self) -> Self {
        if self.lower > 0.0 || self.upper < 0.0 {
            Interval::new(1.0 / self.upper, 1.0 / self.lower)
        } else {
            f32::NAN.into()
        }
    }
    /// Calculates the minimum of two intervals
    ///
    /// Returns both the result and a [`Choice`] indicating whether one side is
    /// always less than the other.
    ///
    /// If either side is `NAN`, returns the `NAN` interval and `Choice::Both`.
    pub fn min_choice(self, rhs: Self) -> (Self, Choice) {
        if self.has_nan() || rhs.has_nan() {
            return (f32::NAN.into(), Choice::Both);
        }
        let choice = if self.upper < rhs.lower {
            Choice::Left
        } else if rhs.upper < self.lower {
            Choice::Right
        } else {
            Choice::Both
        };
        (
            Interval::new(self.lower.min(rhs.lower), self.upper.min(rhs.upper)),
            choice,
        )
    }
    /// Calculates the maximum of two intervals
    ///
    /// Returns both the result and a [`Choice`] indicating whether one side is
    /// always greater than the other.
    ///
    /// If either side is `NAN`, returns the `NAN` interval and `Choice::Both`.
    pub fn max_choice(self, rhs: Self) -> (Self, Choice) {
        if self.has_nan() || rhs.has_nan() {
            return (f32::NAN.into(), Choice::Both);
        }
        let choice = if self.lower > rhs.upper {
            Choice::Left
        } else if rhs.lower > self.upper {
            Choice::Right
        } else {
            Choice::Both
        };
        (
            Interval::new(self.lower.max(rhs.lower), self.upper.max(rhs.upper)),
            choice,
        )
    }

    /// Returns the midpoint of the interval
    pub fn midpoint(self) -> f32 {
        (self.lower + self.upper) / 2.0
    }

    /// Splits the interval at the midpoint
    ///
    /// ```
    /// # use fidget::eval::types::Interval;
    /// let a = Interval::new(0.0, 1.0);
    /// let (lo, hi) = a.split();
    /// assert_eq!(lo, Interval::new(0.0, 0.5));
    /// assert_eq!(hi, Interval::new(0.5, 1.0));
    /// ```
    pub fn split(self) -> (Self, Self) {
        let mid = self.midpoint();
        (
            Interval::new(self.lower, mid),
            Interval::new(mid, self.upper),
        )
    }

    /// Linear interpolation from `lower` to `upper`
    ///
    /// ```
    /// # use fidget::eval::types::Interval;
    /// let a = Interval::new(0.0, 2.0);
    /// assert_eq!(a.lerp(0.5), 1.0);
    /// assert_eq!(a.lerp(0.75), 1.5);
    /// assert_eq!(a.lerp(2.0), 4.0);
    /// ```
    pub fn lerp(self, frac: f32) -> f32 {
        self.lower * (1.0 - frac) + self.upper * frac
    }

    /// Calculates the width of the interval
    ///
    /// ```
    /// # use fidget::eval::types::Interval;
    /// let a = Interval::new(2.0, 3.0);
    /// assert_eq!(a.width(), 1.0);
    /// let b = Interval::new(2.0, 5.0);
    /// assert_eq!(b.width(), 3.0);
    /// ```
    pub fn width(self) -> f32 {
        self.upper - self.lower
    }

    /// Checks that the two values are roughly equal, panicking otherwise
    #[cfg(test)]
    pub(crate) fn compare_eq(&self, other: Self) {
        let d = (self.lower - other.lower)
            .abs()
            .max((self.upper - other.upper).abs());
        if d >= 1e-6 {
            panic!("lhs != rhs ({self:?} != {other:?})");
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower, self.upper)
    }
}

impl From<[f32; 2]> for Interval {
    fn from(i: [f32; 2]) -> Interval {
        Interval::new(i[0], i[1])
    }
}

impl From<f32> for Interval {
    fn from(f: f32) -> Self {
        Interval::new(f, f)
    }
}

impl std::ops::Add<Interval> for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Interval::new(self.lower + rhs.lower, self.upper + rhs.upper)
    }
}

impl std::ops::Mul<Interval> for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self.has_nan() || rhs.has_nan() {
            return f32::NAN.into();
        }
        let mut out = [0.0; 4];
        let mut k = 0;
        for i in [self.lower, self.upper] {
            for j in [rhs.lower, rhs.upper] {
                out[k] = i * j;
                k += 1;
            }
        }
        let mut lower = out[0];
        let mut upper = out[0];
        for &v in &out[1..] {
            lower = lower.min(v);
            upper = upper.max(v);
        }
        Interval::new(lower, upper)
    }
}

impl std::ops::Div<Interval> for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if self.has_nan() {
            return f32::NAN.into();
        }
        if rhs.lower > 0.0 || rhs.upper < 0.0 {
            let mut out = [0.0; 4];
            let mut k = 0;
            for i in [self.lower, self.upper] {
                for j in [rhs.lower, rhs.upper] {
                    out[k] = i / j;
                    k += 1;
                }
            }
            let mut lower = out[0];
            let mut upper = out[0];
            for &v in &out[1..] {
                lower = lower.min(v);
                upper = upper.max(v);
            }
            Interval::new(lower, upper)
        } else {
            f32::NAN.into()
        }
    }
}

impl std::ops::Sub<Interval> for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Interval::new(self.lower - rhs.upper, self.upper - rhs.lower)
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Interval::new(-self.upper, -self.lower)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_interval() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(0.5, 1.5);
        let (v, c) = a.min_choice(b);
        assert_eq!(v, [0.0, 1.0].into());
        assert_eq!(c, Choice::Both);
    }
}
