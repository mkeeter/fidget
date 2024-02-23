//! Implementation for various assemblers on the `x86_64` platform
//!
//! We dedicate 12 registers (`xmm4-16`) to tape data storage, meaning the input
//! tape must be planned with a <= 12 register limit; any spills will live on
//! the stack.
//!
//! Right now, we never call anything, so don't worry about saving stuff.
//!
//! Within a single operation, you'll often need to make use of scratch
//! registers.  `xmm0` is used when loading immediates, and should not be used
//! as a scratch register (this is the `IMM_REG` constant).  `xmm1-3` are all
//! available.

/// We use `xmm4-16` (all caller-saved) for graph variables
pub const REGISTER_LIMIT: usize = 12;
/// `xmm0` is used for immediates
pub const IMM_REG: u8 = 0;
/// `xmm1-3` are available for use as temporaries.
pub const OFFSET: u8 = 4;

pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;
