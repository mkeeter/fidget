/// Represents a single choice made at a min/max node.
///
/// Explicitly stored in a `u8` so that this can be written by JIT functions,
/// which have no notion of Rust enums.
///
/// Note that this is a bitfield such that
/// ```text
/// Choice::Both = Choice::Left | Choice::Right
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Choice {
    Unknown = 0,
    Left = 1,
    Right = 2,
    Both = 3,
}

pub trait Simplify {
    fn simplify(&self, choices: &[Choice]) -> Self;
}
