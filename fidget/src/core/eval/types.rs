//! Custom types used during evaluation
use crate::eval::Choice;

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

/// Represents a range, with conservative calculations to guarantee that it
/// always contains the actual value.
///
/// # Warning
/// This implementation does not set rounding modes, so it may not be _perfect_.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Interval {
    lower: f32,
    upper: f32,
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
        assert!(upper >= lower || (lower.is_nan() && upper.is_nan()));
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
            std::f32::NAN.into()
        } else {
            Interval::new(0.0, self.lower.abs().max(self.upper.abs()).powi(2))
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
                std::f32::NAN.into()
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
            std::f32::NAN.into()
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
            return (std::f32::NAN.into(), Choice::Both);
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
            return (std::f32::NAN.into(), Choice::Both);
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
            return std::f32::NAN.into();
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
            return std::f32::NAN.into();
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
            std::f32::NAN.into()
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
