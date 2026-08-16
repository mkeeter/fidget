use crate::vm::Choice;

/// Extension trait for tape operations on floats
pub trait FloatExt: Sized {
    /// Computes the maximum of two values and a choice
    ///
    /// Unlike `f32::max`, this selects `NAN` when either side is `NAN`
    ///
    /// ```
    /// use fidget_core::{vm::Choice, types::FloatExt};
    ///
    /// let (v, c) = 12.0.max_choice(10.0);
    /// assert_eq!(v, 12.0);
    /// assert_eq!(c, Choice::Left);
    ///
    /// let (v, c) = 1.0.max_choice(2.0);
    /// assert_eq!(v, 2.0);
    /// assert_eq!(c, Choice::Right);
    ///
    /// let (v, c) = 1.0.max_choice(f32::NAN);
    /// assert!(v.is_nan());
    /// assert_eq!(c, Choice::Both);
    /// ```
    fn max_choice(self, other: Self) -> (Self, Choice);

    /// Computes the minimum of two values and a choice
    ///
    /// Unlike `f32::min`, this selects `NAN` when either side is `NAN`
    ///
    /// ```
    /// use fidget_core::{vm::Choice, types::FloatExt};
    ///
    /// let (v, c) = 12.0.min_choice(10.0);
    /// assert_eq!(v, 10.0);
    /// assert_eq!(c, Choice::Right);
    ///
    /// let (v, c) = 1.0.min_choice(2.0);
    /// assert_eq!(v, 1.0);
    /// assert_eq!(c, Choice::Left);
    ///
    /// let (v, c) = 1.0.min_choice(f32::NAN);
    /// assert!(v.is_nan());
    /// assert_eq!(c, Choice::Both);
    /// ```
    fn min_choice(self, other: Self) -> (Self, Choice);

    /// Computes the logical AND between two values
    fn and_choice(self, other: Self) -> (Self, Choice);

    /// Computes the logical OR between two values
    fn or_choice(self, other: Self) -> (Self, Choice);

    /// Signed comparison of two values
    fn compare(self, other: Self) -> Self;

    /// Pseudo-random number generation
    fn rand(self) -> f32;

    /// Pseudo-random number mixing
    fn mix(self, other: f32) -> f32;
}

impl FloatExt for f32 {
    #[inline]
    fn compare(self, other: Self) -> Self {
        self.partial_cmp(&other)
            .map(|c| c as i8 as f32)
            .unwrap_or(f32::NAN)
    }

    #[inline]
    fn max_choice(self, other: Self) -> (Self, Choice) {
        if self > other {
            (self, Choice::Left)
        } else if other > self {
            (other, Choice::Right)
        } else {
            (
                if self.is_nan() || other.is_nan() {
                    f32::NAN
                } else {
                    other
                },
                Choice::Both,
            )
        }
    }

    #[inline]
    fn min_choice(self, other: Self) -> (Self, Choice) {
        if self < other {
            (self, Choice::Left)
        } else if other < self {
            (other, Choice::Right)
        } else {
            (
                if self.is_nan() || other.is_nan() {
                    f32::NAN
                } else {
                    other
                },
                Choice::Both,
            )
        }
    }

    #[inline]
    fn and_choice(self, other: Self) -> (Self, Choice) {
        if self == 0.0 {
            (self, Choice::Left)
        } else {
            (other, Choice::Right)
        }
    }

    #[inline]
    fn or_choice(self, other: Self) -> (Self, Choice) {
        if self != 0.0 {
            (self, Choice::Left)
        } else {
            (other, Choice::Right)
        }
    }

    #[inline]
    fn rand(self) -> Self {
        crate::rng::rand(self.to_bits())
    }

    #[inline]
    fn mix(self, other: Self) -> Self {
        f32::from_bits(crate::rng::mix(self.to_bits(), other.to_bits()))
    }
}
