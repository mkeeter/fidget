//! Compiler infrastructure
//!
//! The Fidget compiler operates in several stages:
//! - A math graph (specified as a [`Context`](crate::Context) and
//!   [`Node`](crate::context::Node)) is flattened into an [`SsaTape`], i.e. a
//!   set of operations in single-static assignment form.
//! - The [`SsaTape`] goes through [register allocation](RegisterAllocator) and
//!   becomes a [`RegTape`], planned with some number of registers.

mod alloc;
pub use alloc::RegisterAllocator;

mod op;

mod lru;
pub(crate) use lru::Lru;
pub use op::{RegOp, RegOpDiscriminants, RegOpDiscriminantsIter, SsaOp};

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
