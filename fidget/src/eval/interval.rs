use crate::eval::EvalMath;

#[derive(Copy, Clone, Debug)]
pub struct Interval {
    pub lower: f32,
    pub upper: f32,
}

impl EvalMath for Interval {
    fn abs(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval {
                    lower: 0.0,
                    upper: self.upper.max(-self.lower),
                }
            } else {
                Interval {
                    lower: -self.upper,
                    upper: -self.lower,
                }
            }
        } else {
            self
        }
    }
    fn sqrt(self) -> Self {
        if self.lower < 0.0 {
            if self.upper > 0.0 {
                Interval {
                    lower: 0.0,
                    upper: self.upper.sqrt(),
                }
            } else {
                std::f32::NAN.into()
            }
        } else {
            Interval {
                lower: self.lower.sqrt(),
                upper: self.upper.sqrt(),
            }
        }
    }
    fn recip(self) -> Self {
        todo!()
    }
    fn min(self, rhs: Self) -> Self {
        Interval {
            lower: self.lower.min(rhs.lower),
            upper: self.upper.min(rhs.upper),
        }
    }
    fn max(self, rhs: Self) -> Self {
        Interval {
            lower: self.lower.max(rhs.lower),
            upper: self.upper.max(rhs.upper),
        }
    }
}

impl From<f32> for Interval {
    fn from(f: f32) -> Self {
        Interval { lower: f, upper: f }
    }
}

impl std::ops::Add<Interval> for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Interval {
            lower: self.lower + rhs.lower,
            upper: self.upper + rhs.upper,
        }
    }
}

impl std::ops::Mul<Interval> for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
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
        Interval { lower, upper }
    }
}

impl std::ops::Sub<Interval> for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Interval {
            lower: self.lower - rhs.upper,
            upper: self.upper - rhs.lower,
        }
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Interval {
            lower: -self.upper,
            upper: -self.lower,
        }
    }
}
