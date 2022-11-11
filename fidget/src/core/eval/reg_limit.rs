/// Helper trait to indicate the register limit associated with a
/// [`Tape`](crate::eval::tape::Tape) or evaluator family.
///
/// This is needed because const generics aren't quite powerful enough to do
/// everything at compile-time.
pub trait RegLimit {
    fn reg_limit() -> u8;
}

/// Simple struct to translate from a const generic to a [`RegLimit`](RegLimit)
pub struct ConstRegLimit<const N: u8>;
impl<const N: u8> RegLimit for ConstRegLimit<N> {
    fn reg_limit() -> u8 {
        N
    }
}
