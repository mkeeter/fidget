//! Compiler infrastructure
//!
//! The Fidget compiler operates in several stages:
//! - A math graph (specified as a (`Context`)[crate::Context] and
//!   (`Node`)[crate::context::Node]) is flattened into an `SsaTape`, i.e. a set
//!   of operations in single-static assignment form.
//! - An `SsaTape` goes through register allocation and becomes a `RegTape`,
//!   planned with some number of registers.

mod alloc;
pub use alloc::RegisterAllocator;

mod op;
pub use op::TapeOp;

mod lru;
pub(crate) use lru::Lru;

/// Operation used in register-allocated tapes
///
/// We have a maximum of 256 registers, though some tapes (e.g. ones targeting
/// physical hardware) may choose to use fewer.
pub type RegOp = TapeOp<u8>;

/// Operation used in single-static-assignment (SSA) tapes
///
/// Pseudo-registers are never reused, and we are limited to `u32::MAX` clauses.
pub type SsaOp = TapeOp<u32>;

mod reg_tape;
mod ssa_tape;

pub use reg_tape::RegTape;
pub use ssa_tape::SsaTape;

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_vm_op_size() {
        assert_eq!(std::mem::size_of::<RegOp>(), 8);
        assert_eq!(std::mem::size_of::<SsaOp>(), 16);
    }
}
