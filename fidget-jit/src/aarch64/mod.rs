//! Implementation for various assemblers on the `aarch64` platform
//!
//! We dedicate 24 registers to tape data storage:
//! - Floating point registers `s8-15` (callee-saved, but only the lower 64
//!   bits)
//! - Floating-point registers `s16-31` (caller-saved)
//!
//! This means that the input tape must be planned with a <= 24 register limit;
//! any spills will live on the stack.
//!
//! Right now, we never call anything, so don't worry about saving stuff.
//!
//! Within a single operation, you'll often need to make use of scratch
//! registers.  `s3` / `v3` is used when loading immediates, and should not be
//! used as a scratch register (this is the `IMM_REG` constant).  `s4-7`/`v4-7`
//! are all available, and are callee-saved.
//!
//! For general-purpose registers, `x9-15` (also called `w9-15`) are reasonable
//! choices; they are caller-saved, so we can trash them at will.

/// We can use registers `v8-15` (callee saved) and `v16-31` (caller saved)
pub const REGISTER_LIMIT: usize = 24;
/// `v3` is used for immediates, because `v0-2` contain inputs
pub const IMM_REG: u8 = 3;
/// `v4-7` are used for as temporary variables
pub const OFFSET: u8 = 8;

pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;
