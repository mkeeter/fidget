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
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    /// This choice has not yet been assigned
    ///
    /// A value of `Unknown` is invalid after evaluation
    Unknown = 0,

    /// The operation always picks the left-hand input
    Left = 1,

    /// The operation always picks the right-hand input
    Right = 2,

    /// The operation may pick either input
    Both = 3,
}

impl std::ops::BitOrAssign<Choice> for Choice {
    fn bitor_assign(&mut self, other: Self) {
        *self = match (*self as u8) | (other as u8) {
            0 => Self::Unknown,
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Both,
            _ => unreachable!(),
        }
    }
}
