/// Custom types used during evaluation
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
    #[inline]
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(upper >= lower || (lower.is_nan() && upper.is_nan()));
        Self { lower, upper }
    }
    #[inline]
    pub fn lower(&self) -> f32 {
        self.lower
    }
    #[inline]
    pub fn upper(&self) -> f32 {
        self.upper
    }
    pub fn has_nan(&self) -> bool {
        self.lower.is_nan() || self.upper.is_nan()
    }
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
    pub fn recip(self) -> Self {
        if self.lower > 0.0 || self.upper < 0.0 {
            Interval::new(1.0 / self.upper, 1.0 / self.lower)
        } else {
            std::f32::NAN.into()
        }
    }
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
