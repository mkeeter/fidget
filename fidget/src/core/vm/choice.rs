/// A single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
///
/// Note that this is a bitfield such that
/// ```rust
/// # use fidget::vm::Choice;
/// # assert!(
/// Choice::Both as u8 == Choice::Left as u8 | Choice::Right as u8
/// # );
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Choice {
    /// This operation has not picked either left or right-hand input
    None = 0,

    /// The operation always picks the left-hand input
    Left = 1,

    /// The operation always picks the right-hand input
    Right = 2,

    /// The operation may pick either input
    Both = 3,
}

impl Default for Choice {
    fn default() -> Self {
        Self::None
    }
}

impl std::ops::BitOrAssign<Choice> for Choice {
    fn bitor_assign(&mut self, other: Self) {
        *self = match (*self as u8) | (other as u8) {
            0 => Self::None,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Both,
            _ => unreachable!(),
        }
    }
}

impl std::ops::Not for Choice {
    type Output = Choice;
    fn not(self) -> Self {
        match self {
            Self::None => Self::Both,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Both => Self::None,
        }
    }
}

impl std::ops::BitAndAssign<Choice> for Choice {
    fn bitand_assign(&mut self, other: Self) {
        *self = (*self) & other
    }
}

impl std::ops::BitAnd<Choice> for Choice {
    type Output = Choice;
    fn bitand(self, other: Self) -> Self::Output {
        match (self as u8) & (other as u8) {
            0 => Self::None,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Both,
            _ => unreachable!(),
        }
    }
}
