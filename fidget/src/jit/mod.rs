//! Infrastructure for compiling down to native machine code
//!
//! # So you want to write some assembly?
//! There are four flavors of functions, with different signatures
//! - Point-wise evaluation passes 3x `f32` in `s0-2`, and a choice array in
//!   `x0`
//! - Float slice evaluation passes 3x `*const f32` in `x0-2`, and an output
//!   array `*mut f32` in `x3`.  Each pointer represents 4x `f32`
//! - Interval evaluation passes 6x `f32` in `s0-5`, a choice array `*mut u8`
//!   in `x0`, and returns 2x `f32` in `s0-1`.  Each pair represents an
//!   interval.  During the function prelude, `s0-5` are packed into `v0-2`
//! - Gradient slice evaluation passes 3x `f32` in `s0-2`, and returns outputs
//!   in `s0-3`, which represent `[v, dx, dy, dz]`.
//!
//! ## In this house, we obey the A64 calling convention.
//! We dedicate 24 registers to tape data storage:
//! - Floating point registers `s8-15` (callee-saved, but only the lower 64
//!   bits)
//! - Floating-point registers `s16-31` (caller-saved)
//!
//! Right now, we never call anything, so don't worry about saving stuff.
//!
//! Register format depends on evaluation flavor:
//! - Point-wise evaluation uses `sX`, i.e. a single float
//! - Float slice evaluation uses `vX.s4`, i.e. an array of 4 floats
//! - Interval evaluation uses the lower two floats in `vX.s4`: `s[0]` is the
//!   lower bound, and `s[1]` is the upper.
//! - Gradient evaluation uses `vX.s4` as `[v, dx, dy, dz]`
//!
//! ## Scratch registers
//! Within a single operation, you'll often need to make use of scratch
//! registers.  `s3` / `v3` is used when loading immediates, and should not be
//! used as a scratch register.  `s4-7`/`v4-7` are all available.
//!
//! For general-purpose registers, `x9-15` (also called `w9-15`) are reasonable
//! choices.

mod mmap;

use dynasmrt::{aarch64::Aarch64Relocation, dynasm, DynasmApi, VecAssembler};

use std::sync::Arc;

use crate::{
    asm::AsmOp,
    eval::{
        float_slice::FloatSliceEvalT,
        grad::{Grad, GradEvalT},
        interval::{Interval, IntervalEvalT},
        point::PointEvalT,
        Choice, EvalFamily,
    },
    jit::mmap::Mmap,
    tape::Tape,
};

/// Number of registers available when executing natively
///
/// We can use registers v8-v15 (callee saved) and v16-v31 (caller saved)
pub const REGISTER_LIMIT: u8 = 24;

/// Offset before the first useable register
const OFFSET: u8 = 8;

/// Register written to by `CopyImm`
///
/// `IMM_REG` is selected to avoid scratch registers used by other
/// functions, e.g. interval mul / min / max
const IMM_REG: u8 = 3;

/// Converts from a tape-local register to an AArch64 register
///
/// Tape-local registers are in the range `0..REGISTER_LIMIT`, while ARM
/// registers have an offset (based on calling convention).
///
/// This uses `wrapping_add` to support immediates, which are loaded into an ARM
/// register below `OFFSET` (which is "negative" from the perspective of this
/// function).
fn reg(r: u8) -> u32 {
    let out = r.wrapping_add(OFFSET) as u32;
    assert!(out < 32);
    out
}

const CHOICE_LEFT: u32 = Choice::Left as u32;
const CHOICE_RIGHT: u32 = Choice::Right as u32;
const CHOICE_BOTH: u32 = Choice::Both as u32;

trait AssemblerT {
    fn init() -> Self;
    fn build_load(&mut self, dst_reg: u8, src_mem: u32);
    fn build_store(&mut self, dst_mem: u32, src_reg: u8);

    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8);
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Loads an immediate into a register, returning that register
    fn load_imm(&mut self, imm: f32) -> u8;

    fn finalize(self, out_reg: u8) -> Vec<u8>;
}

/////////////////////////////////////////////////////////////////////////////////////////

struct AssemblerData<T> {
    ops: VecAssembler<Aarch64Relocation>,

    /// Current offset of the stack pointer, in bytes
    mem_offset: usize,

    _p: std::marker::PhantomData<*const T>,
}

impl<T> AssemblerData<T> {
    fn check_stack(&mut self, mem_slot: u32) -> u32 {
        assert!(mem_slot >= REGISTER_LIMIT as u32);
        let mem = (mem_slot as usize - REGISTER_LIMIT as usize)
            * std::mem::size_of::<T>();
        // Round up to the nearest multiple of 16 bytes, for alignment
        let mem_end = (((mem + std::mem::size_of::<T>()) + 15) / 16) * 16;

        if mem_end > self.mem_offset {
            let addr = u32::try_from(mem_end - self.mem_offset).unwrap();
            dynasm!(self.ops
                ; sub sp, sp, #(addr)
            );
            self.mem_offset = mem_end;
        }

        // Return the offset of the given slot, computed based on the new stack
        // pointer location in memory.
        u32::try_from(self.mem_offset - (mem + std::mem::size_of::<T>()))
            .unwrap()
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

struct PointAssembler(AssemblerData<f32>);

impl AssemblerT for PointAssembler {
    fn init() -> Self {
        let mut ops = VecAssembler::new(0);
        dynasm!(ops
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

        Self(AssemblerData {
            ops,
            mem_offset: 0,
            _p: std::marker::PhantomData,
        })
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(src_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; ldr S(reg(dst_reg)), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(dst_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; str S(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; fmov S(reg(out_reg)), S(src_arg as u32));
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmov S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fneg S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fabs S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov s7, #1.0
            ; fdiv S(reg(out_reg)), s7, S(reg(lhs_reg))
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fsqrt S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(lhs_reg)))
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fadd S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fsub S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fdiv S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; ldrb w14, [x0]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi #20 // -> RHS
            ; b.gt #28 // -> LHS

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmax S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_BOTH
            ; b #24 // -> end

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; b #12

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            // fall-through to end

            // <- end
            ; strb w14, [x0], #1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; ldrb w14, [x0]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi #20
            ; b.gt #28

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmin S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_BOTH
            ; b #24 // -> end

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; b #12

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            // fall-through to end

            // <- end
            ; strb w14, [x0], #1 // post-increment
        )
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; fmov S(IMM_REG as u32), w9
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self, out_reg: u8) -> Vec<u8> {
        dynasm!(self.0.ops
            // Prepare our return value
            ; fmov  s0, S(reg(out_reg))
            // Restore stack space used for spills
            ; add   sp, sp, #(self.0.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        self.0.ops.finalize().unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Alright, here's the plan.
///
/// We're calling a function of the form
/// ```
/// # type IntervalFn =
/// extern "C" fn([f32; 2], [f32; 2], [f32; 2], *mut u8) -> [f32; 2];
/// ```
///
/// The first three arguments are `x`, `y`, and `z` intervals.  They come packed
/// into `s0-5`, and we shuffle them into SIMD registers `V0.2S`, `V1.2S`, and
/// `V2.2s` respectively.
///
/// The last argument is a pointer to the `choices` array, which is populated
/// by `min` and `max` opcodes.  It comes in the `x0` register, which is
/// unchanged by our function.
///
/// During evaluation, each SIMD register stores an interval.  `s[0]` is the
/// lower bound of the interval and `s[1]` is the upper bound.
///
/// The input tape must be planned with a <= 24 register limit.  We use hardware
/// `V8.2S` through `V32.2S` to store our tape registers, and put everything
/// else on the stack.
///
/// `V4.2S` through `V7.2S` are used for scratch values within a single opcode
/// (e.g. storing intermediate values when calculating `min` or `max`).
///
/// In general, expect to use `v4` and `v5` for intermediate (float) values,
/// and `[x,w]15` for intermediate integer values.  These are all caller-saved,
/// so we can trash them at will.
struct IntervalAssembler(AssemblerData<[f32; 2]>);

impl AssemblerT for IntervalAssembler {
    fn init() -> Self {
        let mut ops = dynasmrt::VecAssembler::new(0);
        dynasm!(ops
            // Preserve frame and link register
            ; stp   x29, x30, [sp, #-16]!
            // Preserve sp
            ; mov   x29, sp
            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, #-16]!
            ; stp   d10, d11, [sp, #-16]!
            ; stp   d12, d13, [sp, #-16]!
            ; stp   d14, d15, [sp, #-16]!

            // Arguments are passed in S0-5; collect them into V0-1
            ; mov v0.s[1], v1.s[0]
            ; mov v1.s[0], v2.s[0]
            ; mov v1.s[1], v3.s[0]
            ; mov v2.s[0], v4.s[0]
            ; mov v2.s[1], v5.s[0]
        );

        Self(AssemblerData {
            ops,
            mem_offset: 0,
            _p: std::marker::PhantomData,
        })
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(src_mem);
        assert!(sp_offset <= 32768);
        dynasm!(self.0.ops ; ldr D(reg(dst_reg)), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(dst_mem);
        assert!(sp_offset <= 32768);
        dynasm!(self.0.ops ; str D(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; fmov D(reg(out_reg)), D(src_arg as u32));
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmov D(reg(out_reg)), D(reg(lhs_reg)))
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fneg V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2
        )
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store lhs < 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4

            // Store abs(lhs) in V(reg(out_reg))
            ; fabs V(reg(out_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> upper_lz

            // Check whether lhs.lower < 0
            ; tst x15, #0x1

            // otherwise, we're good; return the original
            ; b.eq #20 // -> end

            // if lhs.lower < 0, then the output is
            //  [0.0, max(abs(lower, upper))]
            ; movi d4, #0
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; fmov D(reg(out_reg)), d4
            // Fall through to do the swap

            // <- upper_lz
            // if upper < 0
            //   return [-upper, -lower]
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0.ops
            // Check whether lhs.lower > 0.0
            ; fcmp S(reg(lhs_reg)), 0.0
            ; b.gt #32 // -> okay

            // Check whether lhs.upper < 0.0
            ; mov s4, V(reg(lhs_reg)).s[1]
            ; fcmp s4, 0.0
            ; b.mi #20 // -> okay

            // Bad case: the division spans 0, so return NaN
            ; movz w15, #(nan_u32 >> 16), lsl 16
            ; movk w15, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w15
            ; b #20 // -> end

            // <- okay
            ; fmov s4, #1.0
            ; dup v4.s2, v4.s[0]
            ; fdiv V(reg(out_reg)).s2, v4.s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0.ops
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne #40 // -> upper_lz

            ; tst x15, #0x1
            ; b.ne #12 // -> lower_lz

            // Happy path
            ; fsqrt V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; b #36 // -> end

            // <- lower_lz
            ; mov v4.s[0], V(reg(lhs_reg)).s[1]
            ; fsqrt s4, s4
            ; movi D(reg(out_reg)), #0
            ; mov V(reg(out_reg)).s[1], v4.s[0]
            ; b #16

            // <- upper_lz
            ; movz w9, #(nan_u32 >> 16), lsl 16
            ; movk w9, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w9

            // <- end
        )
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4
            ; fmul V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper <= 0.0
            ; tst x15, #0x1_0000_0000
            ; b.ne #28 // -> swap

            // Test whether lhs.lower <= 0.0
            ; tst x15, #0x1
            ; b.eq #24 // -> end

            // If the input interval straddles 0, then the
            // output is [0, max(lower**2, upper**2)]
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; movi D(reg(out_reg)), #0
            ; mov V(reg(out_reg)).s[1], v4.s[0]
            ; b #8 // -> end

            // <- swap
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fadd V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; rev64 v4.s2, V(reg(rhs_reg)).s2
            ; fsub V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, v4.s2
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Set up v4 to contain
            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
            // and v5 to contain
            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
            //
            // Multiplying them out will hit all four possible
            // combinations; then we extract the min and max
            // with vector-reducing operations
            ; rev64 v4.s2, V(reg(lhs_reg)).s2
            ; mov v4.d[1], V(reg(lhs_reg)).d[0]
            ; dup v5.d2, V(reg(rhs_reg)).d[0]

            ; fmul v4.s4, v4.s4, v5.s4
            ; fminnmv S(reg(out_reg)), v4.s4
            ; fmaxnmv s5, v4.s4
            ; mov V(reg(out_reg)).s[1], v5.s[0]
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0.ops
            // Store rhs.lower > 0.0 in x15, then check rhs.lower > 0
            ; fcmp S(reg(rhs_reg)), #0.0
            ; b.gt #32 // -> happy

            // Store rhs.upper < 0.0 in x15, then check rhs.upper < 0
            ; mov s4, V(reg(rhs_reg)).s[1]
            ; fcmp s4, #0.0
            ; b.lt #20

            // Sad path: rhs spans 0, so the output includes NaN
            ; movz w9, #(nan_u32 >> 16), lsl 16
            ; movk w9, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w9
            ; b #32 // -> end

            // >happy:
            // Set up v4 to contain
            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
            // and v5 to contain
            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
            //
            // Dividing them out will hit all four possible
            // combinations; then we extract the min and max
            // with vector-reducing operations
            ; rev64 v4.s2, V(reg(lhs_reg)).s2
            ; mov v4.d[1], V(reg(lhs_reg)).d[0]
            ; dup v5.d2, V(reg(rhs_reg)).d[0]

            ; fdiv v4.s4, v4.s4, v5.s4
            ; fminnmv S(reg(out_reg)), v4.s4
            ; fmaxnmv s5, v4.s4
            ; mov V(reg(out_reg)).s[1], v5.s[0]

            // >end
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Basically the same as MinRegReg
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5
            ; ldrb w14, [x0]

            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> lhs

            ; tst x15, #0x1
            ; b.eq #28 // -> both

            // LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; b #24 // -> end

            // <- lhs (when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; b #12 // -> end

            // <- both
            ; fmax V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; orr w14, w14, #CHOICE_BOTH

            // <- end
            ; strb w14, [x0], #1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            //  if lhs.upper < rhs.lower
            //      *choices++ |= CHOICE_LEFT
            //      out = lhs
            //  elif rhs.upper < lhs.lower
            //      *choices++ |= CHOICE_RIGHT
            //      out = rhs
            //  else
            //      *choices++ |= CHOICE_BOTH
            //      out = fmin(lhs, rhs)

            // v4 = [lhs.upper, rhs.upper]
            // v5 = [rhs.lower, lhs.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2

            // v5 = [rhs.lower > lhs.upper, lhs.lower > rhs.upper]
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5
            ; ldrb w14, [x0]

            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> rhs

            ; tst x15, #0x1
            ; b.eq #28 // -> both

            // Fallthrough: LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; b #24 // -> end

            // <- rhs (for when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; b #12

            // <- both
            ; fmin V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; orr w14, w14, #CHOICE_BOTH

            // <- end
            ; strb w14, [x0], #1 // post-increment
        )
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; movz w15, #(imm_u32 >> 16), lsl 16
            ; movk w15, #(imm_u32)
            ; dup V(IMM_REG as u32).s2, w15
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self, out_reg: u8) -> Vec<u8> {
        assert!(self.0.mem_offset < 4096);
        dynasm!(self.0.ops
            // Prepare our return value
            ; mov  s0, V(reg(out_reg)).s[0]
            ; mov  s1, V(reg(out_reg)).s[1]
            // Restore stack space used for spills
            ; add   sp, sp, #(self.0.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        self.0.ops.finalize().unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

struct FloatSliceAssembler(AssemblerData<[f32; 4]>);
impl AssemblerT for FloatSliceAssembler {
    fn init() -> Self {
        let mut ops = dynasmrt::VecAssembler::new(0);
        dynasm!(ops
            // Preserve frame and link register
            ; stp   x29, x30, [sp, #-16]!
            // Preserve sp
            ; mov   x29, sp
            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, #-16]!
            ; stp   d10, d11, [sp, #-16]!
            ; stp   d12, d13, [sp, #-16]!
            ; stp   d14, d15, [sp, #-16]!

            // We're actually loading two f32s, but we can pretend they're
            // doubles in order to move 64 bits at a time
            ; ldp d0, d1, [x0]
            ; mov v0.d[1], v1.d[0]
            ; ldp d1, d2, [x1]
            ; mov v1.d[1], v2.d[0]
            ; ldp d2, d3, [x2]
            ; mov v2.d[1], v3.d[0]
        );

        Self(AssemblerData {
            ops,
            mem_offset: 0,
            _p: std::marker::PhantomData,
        })
    }
    /// Reads from `src_mem` to `dst_reg`, using D4 as an intermediary
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(src_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; ldp D(reg(dst_reg)), d4, [x9]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        } else {
            dynasm!(self.0.ops
                ; ldp D(reg(dst_reg)), d4, [sp, #(sp_offset)]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        }
    }

    /// Writes from `src_reg` to `dst_mem`, using D4 as an intermediary
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(dst_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [x9]
            )
        } else {
            dynasm!(self.0.ops
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [sp, #(sp_offset)]
            )
        }
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(src_arg as u32).b16);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16)
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fneg V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fabs V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov s7, #1.0
            ; dup v7.s4, v7.s[0]
            ; fdiv V(reg(out_reg)).s4, v7.s4, V(reg(lhs_reg)).s4
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fsqrt V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmul V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(lhs_reg)).s4
        )
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fadd V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fsub V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmul V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fdiv V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmax V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmin V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }

    /// Loads an immediate into register V4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; dup V(IMM_REG as u32).s4, w9
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self, out_reg: u8) -> Vec<u8> {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x3
            // It's fine to overwrite X at this point in V0, since we're not
            // using it anymore.
            ; mov v0.d[0], V(reg(out_reg)).d[1]
            ; stp D(reg(out_reg)), d0, [x3]

            // Restore stack space used for spills
            ; add   sp, sp, #(self.0.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        self.0.ops.finalize().unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Assembler for automatic differentiation / gradient evaluation
///
/// Each vector register is used to store `[value, dx, dy, dz]`; `value` is
/// in the `s0` position for convenience.
struct GradAssembler(AssemblerData<f32>);

impl AssemblerT for GradAssembler {
    fn init() -> Self {
        let mut ops = VecAssembler::new(0);
        dynasm!(ops
            // Preserve frame and link register
            ; stp   x29, x30, [sp, #-16]!
            // Preserve sp
            ; mov   x29, sp
            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, #-16]!
            ; stp   d10, d11, [sp, #-16]!
            ; stp   d12, d13, [sp, #-16]!
            ; stp   d14, d15, [sp, #-16]!

            // Arguments are passed in S0-2; inject the derivatives here
            ; fmov s6, #1.0
            ; mov v0.S[1], v6.S[0]
            ; mov v1.S[2], v6.S[0]
            ; mov v2.S[3], v6.S[0]
        );

        Self(AssemblerData {
            ops,
            mem_offset: 0,
            _p: std::marker::PhantomData,
        })
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(src_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; ldp D(reg(dst_reg)), d4, [x9]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        } else {
            dynasm!(self.0.ops
                ; ldp D(reg(dst_reg)), d4, [sp, #(sp_offset)]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        }
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.check_stack(dst_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [x9]
            )
        } else {
            dynasm!(self.0.ops
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [sp, #(sp_offset)]
            )
        }
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(src_arg as u32).b16);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16)
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fneg V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        // TODO: use two fcsel instead?
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), 0.0
            ; b.lt #12 // -> neg
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; b #8 // -> end
            // neg:
            ; fneg V(reg(out_reg)).s4, V(reg(lhs_reg)).s4
            // end:
        )
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmul s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; fneg s6, s6
            ; dup v6.s4, v6.s[0]
            ; fdiv v7.s4, V(reg(lhs_reg)).s4, v6.s4
            ; fmov s6, #1.0
            ; fdiv s6, s6, S(reg(lhs_reg))
            ; mov V(reg(out_reg)).b16, v7.b16
            ; mov V(reg(out_reg)).s[0], v6.s[0]
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fsqrt s6, S(reg(lhs_reg))
            ; fmov s7, #2.0
            ; fmul s7, s6, s7
            ; dup v7.s4, v7.s[0]
            ; fdiv V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, v7.s4
            ; mov V(reg(out_reg)).S[0], v6.S[0]
        )
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov s7, #2.0
            ; dup v7.s4, v7.s[0]
            ; fmov s6, #1.0
            ; mov v7.S[0], v6.S[0]
            // At this point, v7.s4 is [2.0, 2.0, 2.0, 1.0]
            ; fmov w9, S(reg(lhs_reg))
            ; dup v6.s4, w9
            // Now, v6.s4 is [v, v, v, v]
            ; fmul V(reg(out_reg)).s4, v6.s4, V(reg(lhs_reg)).s4
            // out is [v*v, v*dx, v*dy, v*dz]
            ; fmul V(reg(out_reg)).s4, v7.s4, V(reg(out_reg)).s4
            // out is [v*v, 2*v*dx, 2*v*dy, 2*v*dz]
        )
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fadd V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fsub V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // v6.s4 = [lhs.v, lhs.v, lhs.v, lhs.v]
            ; dup v6.s4, V(reg(lhs_reg)).s[0]

            // v5 = [lhs.v * rhs.v, lhs.v * rhs.dx, lhs.v * rhs.dy, ...]
            ; fmul v5.s4, v6.s4, V(reg(rhs_reg)).s4

            // s7 = lhs.v * rhs.v (copied from v5.s[0])
            ; fmov s7, s5

            // v6.s4 = [rhs.v, rhs.v, rhs.v, rhs.v]
            ; dup v6.s4, V(reg(rhs_reg)).s[0]

            // v5.s4 = [lhs.v * rhs.v + rhs.v * lhs.v,
            //          lhs.v * rhs.dx + rhs.v * lhs.dx,
            //          lhs.v * rhs.dy + rhs.v * lhs.dy,
            //          lhs.v * rhs.dz + rhs.v * lhs.dz]
            // (i.e. everything is right except out.s[0])
            ; fmla v5.s4, v6.s4, V(reg(lhs_reg)).s4

            // Copy stuff into the output register
            ; mov V(reg(out_reg)).b16, v5.b16
            ; mov V(reg(out_reg)).s[0], v7.s[0]
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov w9, S(reg(rhs_reg))
            ; dup v6.s4, w9
            ; fmul v5.s4, v6.s4, V(reg(lhs_reg)).s4
            // At this point, gradients are of the form
            //      rhs.v * lhs.d

            ; fmov w9, S(reg(lhs_reg))
            ; dup v6.s4, w9
            ; fmls v5.s4, v6.s4, V(reg(lhs_reg)).s4
            // At this point, gradients are of the form
            //      rhs.v * lhs.d - lhs.v * rhs.d

            // Divide by rhs.v**2
            ; fmul s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; fmov w9, s6
            ; dup v6.s4, w9
            ; fdiv v5.s4, V(reg(out_reg)).s4, v6.s4

            // Patch in the actual division value
            ; mov V(reg(out_reg)).b16, v5.b16
            ; fdiv s5, S(reg(lhs_reg)), S(reg(rhs_reg))
            ; mov V(reg(out_reg)).s[0], v5.s[0]
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.gt #12 // -> lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b #8 // -> end
            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            // end:
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.lt #12 // -> lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b #8 // -> end
            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            // end:
        )
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; fmov S(IMM_REG as u32), w9
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self, out_reg: u8) -> Vec<u8> {
        dynasm!(self.0.ops
            // Prepare our return value, writing into s0-3
            ; mov s0, V(reg(out_reg)).s[0]
            ; mov s1, V(reg(out_reg)).s[1]
            ; mov s2, V(reg(out_reg)).s[2]
            ; mov s3, V(reg(out_reg)).s[3]

            // Restore stack space used for spills
            ; add   sp, sp, #(self.0.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        self.0.ops.finalize().unwrap()
    }
}

/// Evaluator for a JIT-compiled function performing gradient evaluation.
pub struct JitGradEval {
    mmap: Arc<Mmap>,
    /// X, Y, Z are passed by value; the output is written to an array of 4
    /// floats (allocated by the caller)
    fn_grad: unsafe extern "C" fn(f32, f32, f32) -> [f32; 4],
}

impl From<Tape> for JitGradEval {
    fn from(t: Tape) -> Self {
        let buf = build_asm_fn::<GradAssembler>(t.iter_asm());
        let mut mmap = Mmap::new(buf.len()).unwrap();
        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            fn_grad: unsafe { std::mem::transmute(ptr) },
        }
    }
}

impl GradEvalT for JitGradEval {
    type Family = JitEvalFamily;
    type Storage = Mmap;

    fn from_tape_give(tape: Tape, prev: Self::Storage) -> Self {
        let buf = build_asm_fn::<GradAssembler>(tape.iter_asm());
        let mut mmap = if buf.len() <= prev.len() {
            prev
        } else {
            Mmap::new(buf.len()).unwrap()
        };

        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            fn_grad: unsafe { std::mem::transmute(ptr) },
        }
    }
    fn take(mut self) -> Option<Self::Storage> {
        let t = Arc::get_mut(&mut self.mmap)?;
        Some(std::mem::take(t))
    }
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> Grad {
        let [v, x, y, z] = unsafe { (self.fn_grad)(x, y, z) };
        Grad::new(v, x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////

fn build_asm_fn<A: AssemblerT>(i: impl Iterator<Item = AsmOp>) -> Vec<u8> {
    let mut asm = A::init();

    for op in i {
        use AsmOp::*;
        match op {
            Load(reg, mem) => {
                asm.build_load(reg, mem);
            }
            Store(reg, mem) => {
                asm.build_store(mem, reg);
            }
            Input(out, i) => {
                asm.build_input(out, i);
            }
            NegReg(out, arg) => {
                asm.build_neg(out, arg);
            }
            AbsReg(out, arg) => {
                asm.build_abs(out, arg);
            }
            RecipReg(out, arg) => {
                asm.build_recip(out, arg);
            }
            SqrtReg(out, arg) => {
                asm.build_sqrt(out, arg);
            }
            CopyReg(out, arg) => {
                asm.build_copy(out, arg);
            }
            SquareReg(out, arg) => {
                asm.build_square(out, arg);
            }
            AddRegReg(out, lhs, rhs) => {
                asm.build_add(out, lhs, rhs);
            }
            MulRegReg(out, lhs, rhs) => {
                asm.build_mul(out, lhs, rhs);
            }
            DivRegReg(out, lhs, rhs) => {
                asm.build_div(out, lhs, rhs);
            }
            SubRegReg(out, lhs, rhs) => {
                asm.build_sub(out, lhs, rhs);
            }
            MinRegReg(out, lhs, rhs) => {
                asm.build_min(out, lhs, rhs);
            }
            MaxRegReg(out, lhs, rhs) => {
                asm.build_max(out, lhs, rhs);
            }
            AddRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_add(out, arg, reg);
            }
            MulRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_mul(out, arg, reg);
            }
            DivRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, arg, reg);
            }
            DivImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, reg, arg);
            }
            SubImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_sub(out, reg, arg);
            }
            SubRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_sub(out, arg, reg);
            }
            MinRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_min(out, arg, reg);
            }
            MaxRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_max(out, arg, reg);
            }
            CopyImm(out, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_copy(out, reg);
            }
        }
    }

    asm.finalize(0)
}

////////////////////////////////////////////////////////////////////////////////

/// Handle owning a JIT-compiled float function
pub struct JitPointEval {
    _mmap: Arc<Mmap>,
    fn_float: unsafe extern "C" fn(f32, f32, f32, *mut u8) -> f32,
}

impl From<Tape> for JitPointEval {
    fn from(t: Tape) -> Self {
        let buf = build_asm_fn::<PointAssembler>(t.iter_asm());
        let mut mmap = Mmap::new(buf.len()).unwrap();
        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            _mmap: Arc::new(mmap),
            fn_float: unsafe { std::mem::transmute(ptr) },
        }
    }
}

impl PointEvalT for JitPointEval {
    type Family = JitEvalFamily;
    fn eval_p(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        choices: &mut [Choice],
    ) -> f32 {
        unsafe { (self.fn_float)(x, y, z, choices.as_mut_ptr() as *mut u8) }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for a JIT-compiled function taking `[f32; 4]` SIMD values
///
/// The lifetime of this `struct` is bound to an `JitFloatSliceFunc`, which owns
/// the underlying executable memory.
pub struct JitFloatSliceEval {
    mmap: Arc<Mmap>,
    fn_vec: unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32),
}

impl From<Tape> for JitFloatSliceEval {
    fn from(t: Tape) -> Self {
        let buf = build_asm_fn::<FloatSliceAssembler>(t.iter_asm());
        let mut mmap = Mmap::new(buf.len()).unwrap();
        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            fn_vec: unsafe { std::mem::transmute(ptr) },
        }
    }
}

impl FloatSliceEvalT for JitFloatSliceEval {
    type Storage = Mmap;
    type Family = JitEvalFamily;

    fn from_tape_give(t: Tape, prev: Self::Storage) -> Self {
        let buf = build_asm_fn::<FloatSliceAssembler>(t.iter_asm());
        let mut mmap = if buf.len() <= prev.len() {
            prev
        } else {
            Mmap::new(buf.len()).unwrap()
        };

        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        JitFloatSliceEval {
            mmap: Arc::new(mmap),
            fn_vec: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(mut self) -> Option<Self::Storage> {
        let t = Arc::get_mut(&mut self.mmap)?;
        Some(std::mem::take(t))
    }

    fn eval_s(&mut self, xs: &[f32], ys: &[f32], zs: &[f32], out: &mut [f32]) {
        let n = [xs.len(), ys.len(), zs.len(), out.len()]
            .into_iter()
            .min()
            .unwrap();

        // Special case for < 4 items, in which case the input slices can't be
        // used as workspace (because we need at least 4x f32)
        if n < 4 {
            let mut x = [0.0; 4];
            let mut y = [0.0; 4];
            let mut z = [0.0; 4];
            for i in 0..4 {
                x[i] = xs.get(i).cloned().unwrap_or(0.0);
                y[i] = ys.get(i).cloned().unwrap_or(0.0);
                z[i] = zs.get(i).cloned().unwrap_or(0.0);
            }
            let mut tmp = [std::f32::NAN; 4];
            unsafe {
                (self.fn_vec)(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    tmp.as_mut_ptr(),
                );
            }
            out[0..n].copy_from_slice(&tmp[0..n]);
        } else {
            let mut i = 0;
            loop {
                assert!(i + 4 <= n);
                unsafe {
                    (self.fn_vec)(
                        xs.as_ptr().add(i),
                        ys.as_ptr().add(i),
                        zs.as_ptr().add(i),
                        out.as_mut_ptr().add(i),
                    )
                }
                i += 4;
                if i == n {
                    break;
                } else if i + 4 > n {
                    i = n - 4;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for a JIT-compiled function taking `[f32; 2]` intervals
#[derive(Clone)]
pub struct JitIntervalEval {
    mmap: Arc<Mmap>,
    fn_interval: unsafe extern "C" fn(
        [f32; 2], // X
        [f32; 2], // Y
        [f32; 2], // Z
        *mut u8,  // choices
    ) -> [f32; 2],
}

unsafe impl Send for JitIntervalEval {}

impl From<Tape> for JitIntervalEval {
    fn from(t: Tape) -> Self {
        let buf = build_asm_fn::<IntervalAssembler>(t.iter_asm());
        let mut mmap = Mmap::new(buf.len()).unwrap();
        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            fn_interval: unsafe { std::mem::transmute(ptr) },
        }
    }
}

/// Handle owning a JIT-compiled interval function
impl IntervalEvalT for JitIntervalEval {
    type Family = JitEvalFamily;
    type Storage = Mmap;

    fn from_tape_give(tape: Tape, prev: Self::Storage) -> Self {
        let buf = build_asm_fn::<IntervalAssembler>(tape.iter_asm());
        let mut mmap = if buf.len() <= prev.len() {
            prev
        } else {
            Mmap::new(buf.len()).unwrap()
        };

        mmap.copy_from_slice(&buf);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            fn_interval: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(mut self) -> Option<Self::Storage> {
        let t = Arc::get_mut(&mut self.mmap)?;
        Some(std::mem::take(t))
    }

    /// Evaluates an interval
    fn eval_i<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        choices: &mut [Choice],
    ) -> Interval {
        let x: Interval = x.into();
        let y: Interval = y.into();
        let z: Interval = z.into();
        let out = unsafe {
            (self.fn_interval)(
                [x.lower(), x.upper()],
                [y.lower(), y.upper()],
                [z.lower(), z.upper()],
                choices.as_mut_ptr() as *mut u8,
            )
        };
        Interval::new(out[0], out[1])
    }
}

////////////////////////////////////////////////////////////////////////////////

pub enum JitEvalFamily {}
impl EvalFamily for JitEvalFamily {
    const REG_LIMIT: u8 = REGISTER_LIMIT;

    type IntervalEval = JitIntervalEval;
    type FloatSliceEval = JitFloatSliceEval;
    type GradEval = JitGradEval;
    type PointEval = JitPointEval;

    fn tile_sizes_3d() -> &'static [usize] {
        &[64, 16]
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_tests!(JitGradEval);
    crate::interval_tests!(JitIntervalEval);
    crate::float_slice_tests!(JitFloatSliceEval);
    crate::point_tests!(JitPointEval);
}
