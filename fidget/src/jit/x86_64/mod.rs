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
pub const REGISTER_LIMIT: u8 = 12;
/// `xmm0` is used for immediates
pub const IMM_REG: u8 = 0;
/// `xmm1-3` are available for use as temporaries.
pub const OFFSET: u8 = 4;

pub use dynasmrt::x64::X64Relocation as Relocation;

pub mod float_slice;
pub mod grad_slice;
pub mod interval;
pub mod point;

fn prepare_stack(&mut self, slot_count: usize) {
    // We always use the stack on x86_64, if only to store X/Y/Z
    let stack_slots = slot_count.saturating_sub(REGISTER_LIMIT as usize);

    // We put X/Y/Z values at the top of the stack, where they can be
    // accessed with `movss [rbp - i*size_of(T)] xmm`.  This frees up the
    // incoming registers (xmm0-2) in the point evaluator.
    let mem = (stack_slots + 4) * std::mem::size_of::<T>();

    // Round up to the nearest multiple of 16 bytes, for alignment
    self.mem_offset = ((mem + 15) / 16) * 16;
    dynasm!(self.ops
        ; sub rsp, self.mem_offset as i32
    );
}
