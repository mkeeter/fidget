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

pub use dynasmrt::aarch64::Aarch64Relocation as Relocation;

/// We can use registers `v8-15` (callee saved) and `v16-31` (caller saved)
pub const REGISTER_LIMIT: u8 = 24;
/// `v3` is used for immediates, because `v0-2` contain inputs
pub const IMM_REG: u8 = 3;
/// `v6` is used for as a blessed scratch register for memory operations
pub const SCRATCH_REG: u8 = 6;
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
/// Leaves `x15` dirty
pub(self) fn set_choice_bit<D: DynasmApi>(d: &mut D, choice: ChoiceIndex) {
    let i = choice.index as u32;
    let b = choice.bit as u32;
    assert!(i + b / 8 < 4096);
    dynasm!(d
        ; ldrb w15, [x2, #(i + b / 8)]
        ; orr x15, x15, #1 << (b % 8)
        ; strb w15, [x2, #(i + b / 8)]
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
    while b >= 8 {
        assert!(i < 4096);
        dynasm!(d ; strb wzr, [x2, #i]);
        b -= 8;
        i += 1;
    }
    // Write the last byte
    assert!(i < 4096);
    dynasm!(d
        ; mov w15, #1 << b
        ; strb w15, [x2, #i]
    );
}

pub(crate) fn build_jump<D: DynasmApi>(d: &mut D) {
    dynasm!(d
        ; ldr x15, [x0, #8]!
        ; br x15
    );
}

/// Prepares the stack pointer
///
/// Returns a memory offset to be used when restoring the stack pointer in
/// `function_exit`.
pub(crate) fn function_entry<T, D: DynasmApi>(
    d: &mut D,
    slot_count: usize,
) -> u32 {
    dynasm!(d
        // Preserve frame and link register
        ; stp   x29, x30, [sp, #-16]!
        // Preserve sp
        ; mov   x29, sp
        // Preserve callee-saved floating-point registers
        ; stp   d8, d9, [sp, #-16]!
        ; stp   d10, d11, [sp, #-16]!
        ; stp   d12, d13, [sp, #-16]!
        ; stp   d14, d15, [sp, #-16]!
    );
    if slot_count < REGISTER_LIMIT as usize {
        return 0;
    }
    let stack_slots = slot_count - REGISTER_LIMIT as usize;
    let mem = (stack_slots + 1) * std::mem::size_of::<T>();

    // Round up to the nearest multiple of 16 bytes, for alignment
    let mem_offset = ((mem + 15) / 16) * 16;
    assert!(mem_offset < 4096);
    dynasm!(d
        ; sub sp, sp, #(mem_offset as u32)
    );
    mem_offset.try_into().unwrap()
}

pub(crate) fn function_exit<D: DynasmApi>(d: &mut D, mem_offset: u32) {
    dynasm!(d
        // This is our finalization code, which happens after all evaluation
        // is complete.
        //
        // Restore stack space used for spills
        ; add   sp, sp, #mem_offset
        // Restore callee-saved floating-point registers
        ; ldp   d14, d15, [sp], #16
        ; ldp   d12, d13, [sp], #16
        ; ldp   d10, d11, [sp], #16
        ; ldp   d8, d9, [sp], #16
        // Restore frame and link register
        ; ldp   x29, x30, [sp], #16
        ; ret
    );
}

pub(crate) fn stack_pos<T>(slot: u32) -> u32 {
    assert!(slot >= REGISTER_LIMIT as u32);
    (slot - REGISTER_LIMIT as u32) * std::mem::size_of::<T>() as u32
}
