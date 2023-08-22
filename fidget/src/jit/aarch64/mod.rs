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

use crate::vm::ChoiceIndex;
use dynasmrt::{dynasm, DynasmApi};

/// We can use registers `v8-15` (callee saved) and `v16-31` (caller saved)
pub const REGISTER_LIMIT: u8 = 24;
/// `v3` is used for immediates, because `v0-2` contain inputs
pub const IMM_REG: u8 = 3;
/// `v4-7` are used for as temporary variables
pub const OFFSET: u8 = 8;

pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;

/// Sets the given choice bit, reading then writing it to memory
///
/// This function does not set choice bit 0, which must be set elsewhere to
/// indicate that a choice is present at this index.
///
/// Requires the beginning of the `choices` array to be in `x2`
///
/// Leaves `x15` and `x14` dirty
pub(self) fn set_choice_bit<D: DynasmApi>(d: &mut D, choice: ChoiceIndex) {
    let i = choice.index as u32;
    let b = choice.bit as u32;
    dynasm!(d
        ; ldr b15, [x2, #(i + b / 8)]
        ; mov x14, #1 << (b % 8)
        ; orr x15, x15, x14
        ; str b15, [x2, #(i + b / 8)]
    );
}

/// Sets the given choice bit, clearing all bits below it
///
/// This function does not set choice bit 0, which must be set elsewhere to
/// indicate that a choice is present at this index.
///
/// Requires the beginning of the `choices` array to be in `x2`
///
/// Leaves `x15` dirty
pub(self) fn set_choice_exclusive<D: DynasmApi>(
    d: &mut D,
    choice: ChoiceIndex,
) {
    let mut i = choice.index as u32;
    let mut b = choice.bit as u32;
    dynasm!(d ; mov w15, #0);
    while b >= 8 {
        dynasm!(d ; str b15, [x2, #i]);
        b -= 8;
        i += 1;
    }
    // Write the last byte
    dynasm!(d
        ; mov w15, #1 << b
        ; str w15, [x2, #i]
    );
}
