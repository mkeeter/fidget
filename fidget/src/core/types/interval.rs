use crate::vm::Choice;

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
        if self.has_nan() {
            f32::NAN.into()
        } else {
            // TODO: make this smarter
            Interval::new(-1.0, 1.0)
        }
    }
    /// Computes the cosine of the interval
    ///
    /// Right now, this always returns the maximum range of `[-1, 1]`
    pub fn cos(self) -> Self {
        if self.has_nan() {
            f32::NAN.into()
        } else {
            // TODO: make this smarter
            Interval::new(-1.0, 1.0)
        }
    }
    /// Computes the tangent of the interval
    ///
    /// Returns the `NAN` interval if the result contains a undefined point
    pub fn tan(self) -> Self {
        let size = self.upper - self.lower;
        if size >= std::f32::consts::PI {
            f32::NAN.into()
        } else {
            let lower = self.lower.tan();
            let upper = self.upper.tan();
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
        Interval::new(self.lower.atan(), self.upper.atan())
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
            Interval::new(self.lower.ln(), self.upper.ln())
        }
    }
    /// Calculates the square root of the interval
    ///
    /// If the interval contains values below 0, returns a `NAN` interval.
    pub fn sqrt(self) -> Self {
        if self.lower < 0.0 {
            f32::NAN.into()
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
    /// # use fidget::types::Interval;
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
    /// # use fidget::types::Interval;
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
    /// # use fidget::types::Interval;
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

    /// Least non-negative remainder
    pub fn rem_euclid(&self, other: Interval) -> Self {
        // TODO optimize this
        Interval::new(0.0, other.abs().upper())
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

impl std::ops::Mul<f32> for Interval {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        if self.has_nan() || rhs.is_nan() {
            f32::NAN.into()
        } else if rhs < 0.0 {
            Interval::new(self.upper * rhs, self.lower * rhs)
        } else {
            Interval::new(self.lower * rhs, self.upper * rhs)
        }
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
