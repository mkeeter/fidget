use crate::eval::Choice;

/// Trait for math operations used during evaluation
pub trait EvalMath:
    Clone
    + Copy
    + From<f32>
    + std::ops::Add<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn min_choice(self, rhs: Self) -> (Self, Choice);
    fn max_choice(self, rhs: Self) -> (Self, Choice);
}

impl EvalMath for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn min_choice(self, rhs: Self) -> (Self, Choice) {
        (f32::min(self, rhs), Choice::Both)
    }
    fn max_choice(self, rhs: Self) -> (Self, Choice) {
        (f32::max(self, rhs), Choice::Both)
    }
}
