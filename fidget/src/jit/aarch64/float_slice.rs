use crate::jit::{
    float_slice::FloatSliceAssembler, mmap::Mmap, reg, Assembler,
    AssemblerData, Error, IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub const SIMD_WIDTH: usize = 4;

/// Assembler for SIMD point-wise evaluation on `aarch64`
///
/// | Argument | Register | Type                       |
/// | ---------|----------|----------------------------|
/// | `vars`   | `x0`     | `*const *const [f32; 4]`   |
/// | `out`    | `x1`     | `*const *mut [f32; 4]`     |
/// | `count`  | `x2`     | `u64`                      |
///
/// The arrays must be an even multiple of 4 floats, since we're using NEON and
/// 128-bit wide operations for everything.
///
/// During evaluation, the following registers are used:
///
/// | Register | Description                                          |
/// |----------|------------------------------------------------------|
/// | `x3`     | Byte offset within input arrays                      |
/// | `x4`     | Staging for loading SIMD values                      |
/// | `v3.s4`  | Immediate value (`IMM_REG`)                          |
/// | `v7.s4`  | Immediate value for recip (1.0)                      |
/// | `w9`     | Staging for loading immediates                       |
/// | `w15`    | Staging to load variables                            |
/// | `x20-23` | Backups for `x0-3` during function calls             |
/// | `x24`    | Function call address                                |
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0x230    | ...          | Register spills live up here                |
/// | 0x228    | ---          | Padding for 16-byte alignment               |
/// |----------|--------------|---------------------------------------------|
/// | 0x220    | `x24`        | Backup for callee-saved register            |
/// | 0x218    | `x23`        |                                             |
/// | 0x210    | `x22`        |                                             |
/// | 0x208    | `x21`        |                                             |
/// | 0x200    | `x20`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x1c0    | `q31`        | During functions calls, caller-saved tape   |
/// | 0x1b0    | `q30`        | registers are saved on the stack            |
/// | 0x1a0    | `q29`        |                                             |
/// | 0x190    | `q28`        |                                             |
/// | 0x180    | `q27`        |                                             |
/// | 0x170    | `q26`        |                                             |
/// | 0x160    | `q25`        |                                             |
/// | 0x150    | `q24`        |                                             |
/// | 0x140    | `q23`        |                                             |
/// | 0x130    | `q22`        |                                             |
/// | 0x120    | `q21`        |                                             |
/// | 0x110    | `q20`        |                                             |
/// | 0x100    | `q19`        |                                             |
/// | 0xf0     | `q18`        |                                             |
/// | 0xe0     | `q17`        |                                             |
/// | 0xd0     | `q16`        |                                             |
/// | 0xc0     | `q15`        | We also have to save callee-saved registers |
/// | 0xb0     | `q14`        | because the callee only saves the lower 64  |
/// | 0xa0     | `q13`        | bits, and we're using all 128               |
/// | 0x90     | `q12`        |                                             |
/// | 0x80     | `q11`        |                                             |
/// | 0x70     | `q10`        |                                             |
/// | 0x60     | `q9`         |                                             |
/// | 0x50     | `q8`         |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x48     | `d15`        | Callee-saved registers                      |
/// | 0x40     | `d14`        |                                             |
/// | 0x38     | `d13`        |                                             |
/// | 0x30     | `d12`        |                                             |
/// | 0x28     | `d11`        |                                             |
/// | 0x20     | `d10`        |                                             |
/// | 0x18     | `d9`         |                                             |
/// | 0x10     | `d8`         |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x8      | `sp` (`x30`) | Stack frame                                 |
/// | 0x0      | `fp` (`x29`) | [current value for sp]                      |
/// ```
const STACK_SIZE: u32 = 0x230;

impl Assembler for FloatSliceAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        out.prepare_stack(slot_count, STACK_SIZE as usize);
        dynasm!(out.ops
            // Preserve frame and link register, and set up the frame pointer
            ; stp   x29, x30, [sp, 0x0]
            ; mov   x29, sp

            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, 0x10]
            ; stp   d10, d11, [sp, 0x20]
            ; stp   d12, d13, [sp, 0x30]
            ; stp   d14, d15, [sp, 0x40]

            // Back up a few callee-saved registers that we use for functions
            // calls. We have to use `str` here because we're outside the range
            // for `stp`, sadly
            //
            // TODO: only do this if we're doing function calls?
            ; str x20, [sp, 0x200]
            ; str x21, [sp, 0x208]
            ; str x22, [sp, 0x210]
            ; str x23, [sp, 0x218]
            ; str x24, [sp, 0x220]

            ; mov x3, 0

            // The loop returns here, and we check whether we need to loop
            ; ->L:
            // Remember, at this point we have
            //  x0: vars input pointer
            //  x1: output array pointer
            //  x2: number of points to evaluate
            //  x3: offset within SIMD arrays
            //
            // We'll be decrementing x2 by 4 here; x1 and x3 are modified in
            // finalize().

            ; cmp x2, 0
            ; b.eq ->E // function exit

            // Loop body: math begins below
        );

        Self(out)
    }

    fn bytes_per_clause() -> usize {
        10
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem) + STACK_SIZE;
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; ldr Q(reg(dst_reg)), [sp, sp_offset]
        )
    }

    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem) + STACK_SIZE;
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; str Q(reg(src_reg)), [sp, sp_offset]
        )
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg < 16384 / 8);
        dynasm!(self.0.ops
            ; ldr x4, [x0, src_arg * 8]
            ; add x4, x4, x3 // apply array offset
            ; ldr Q(reg(out_reg)), [x4]
        );
    }

    fn build_output(&mut self, arg_reg: u8, out_index: u32) {
        assert!(out_index < 16384 / 8);
        dynasm!(self.0.ops
            ; ldr x4, [x1, out_index * 8]
            ; add x4, x4, x3 // apply array offset
            ; str Q(reg(arg_reg)), [x4] // write to the output array
        );
    }

    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_sin(f: f32) -> f32 {
            f.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_cos(f: f32) -> f32 {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_tan(f: f32) -> f32 {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_asin(f: f32) -> f32 {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_acos(f: f32) -> f32 {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_atan(f: f32) -> f32 {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_exp(f: f32) -> f32 {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_ln(f: f32) -> f32 {
            f.ln()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ln);
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
            ; fmov s7, 1.0
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

    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq v6.s4, V(reg(lhs_reg)).s4, V(reg(lhs_reg)).s4
            ; mvn v6.b16, v6.b16

            // Round, then convert back to f32
            ; fcvtms V(reg(out_reg)).s4, V(reg(lhs_reg)).s4
            ; scvtf V(reg(out_reg)).s4, V(reg(out_reg)).s4

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v6.b16
        );
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq v6.s4, V(reg(lhs_reg)).s4, V(reg(lhs_reg)).s4
            ; mvn v6.b16, v6.b16

            // Round, then convert back to f32
            ; fcvtps V(reg(out_reg)).s4, V(reg(lhs_reg)).s4
            ; scvtf V(reg(out_reg)).s4, V(reg(out_reg)).s4

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v6.b16
        );
    }
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq v6.s4, V(reg(lhs_reg)).s4, V(reg(lhs_reg)).s4
            ; mvn v6.b16, v6.b16

            // Round, then convert back to f32
            ; fcvtas V(reg(out_reg)).s4, V(reg(lhs_reg)).s4
            ; scvtf V(reg(out_reg)).s4, V(reg(out_reg)).s4

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v6.b16
        );
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
    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "C" fn float_atan2(y: f32, x: f32) -> f32 {
            y.atan2(x)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, float_atan2);
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
    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fabs v6.s4, V(reg(rhs_reg)).s4
            ; fdiv v7.s4, V(reg(lhs_reg)).s4, v6.s4
            ; frintm v7.s4, v7.s4 // round down
            ; fmul v7.s4, v7.s4, v6.s4
            ; fsub V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, v7.s4
        )
    }
    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        dynasm!(self.0.ops
            ; cmeq v6.s4, V(reg(arg_reg)).s4, 0
            ; fmov S(reg(out_reg)), 1.0
            ; dup V(reg(out_reg)).s4, V(reg(out_reg)).s[0]
            ; and V(reg(out_reg)).b16, V(reg(out_reg)).b16, v6.b16
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; cmeq v6.s4, V(reg(lhs_reg)).s4, 0
            ; mvn v7.b16, v6.b16
            ; and v6.b16, v6.b16, V(reg(lhs_reg)).b16
            ; and v7.b16, v7.b16, V(reg(rhs_reg)).b16
            ; orr V(reg(out_reg)).b16, v6.b16, v7.b16
        );
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; cmeq v6.s4, V(reg(lhs_reg)).s4, 0
            ; mvn v7.b16, v6.b16
            ; and v7.b16, v7.b16, V(reg(lhs_reg)).b16
            ; and v6.b16, v6.b16, V(reg(rhs_reg)).b16
            ; orr V(reg(out_reg)).b16, v6.b16, v7.b16
        );
    }

    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a mask of valid positions (not NAN)
            ; fcmeq v6.S4, V(reg(lhs_reg)).S4, V(reg(lhs_reg)).S4
            ; fcmeq v7.S4, V(reg(rhs_reg)).S4, V(reg(rhs_reg)).S4
            ; and v6.b16, v6.b16, v7.b16

            // Invert to get a mask of NAN position
            ; mvn v6.b16, v6.b16

            // Note the swap here, from LT -> GT
            ; fcmgt v4.S4, V(reg(rhs_reg)).S4, V(reg(lhs_reg)).S4
            ; fcmgt v5.S4, V(reg(lhs_reg)).S4, V(reg(rhs_reg)).S4
            // At this point, out_reg is all 1s where we should put 1.0

            // Build a map of -1.0 positions
            ; fmov s7, -1.0
            ; dup v7.s4, v7.s[0]
            ; and V(reg(out_reg)).B16, v4.B16, v7.B16

            // Build a map of -1.0 positions
            ; fmov s7, 1.0
            ; dup v7.s4, v7.s[0]
            ; and v5.B16, v5.B16, v7.B16
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v5.B16

            // Build a NAN mask
            ; mov w9, f32::NAN.to_bits().into()
            ; dup v7.s4, w9
            ; and v7.b16, v7.b16, v6.b16

            // Apply NAN mask to NAN positions
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v7.b16
        )
    }

    /// Loads an immediate into register V4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        if imm_u32 & 0xFFFF == 0 {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 >> 16, lsl 16
                ; dup V(IMM_REG as u32).s4, w9
            );
        } else if imm_u32 & 0xFFFF_0000 == 0 {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 & 0xFFFF
                ; dup V(IMM_REG as u32).s4, w9
            );
        } else {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 >> 16, lsl 16
                ; movk w9, imm_u32 & 0xFFFF
                ; dup V(IMM_REG as u32).s4, w9
            );
        }
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // update our "items remaining" counter
            ; sub x2, x2, 4 // We handle 4 items at a time

            // Adjust the input array offset amount
            ; add x3, x3, 16

            // Keep looping!
            ; b ->L

            ; ->E:
            // This is our finalization code, which happens after all evaluation
            // is complete.
            //
            // Restore frame and link register
            ; ldp   x29, x30, [sp, 0x0]

            // Restore callee-saved floating-point registers
            ; ldp   d8, d9, [sp, 0x10]
            ; ldp   d10, d11, [sp, 0x20]
            ; ldp   d12, d13, [sp, 0x30]
            ; ldp   d14, d15, [sp, 0x40]

            // Restore callee-saved registers (using `ldr` because we're outside
            // the range for `ldp`).  TODO: only do this if the tape contains
            // function calls?
            ; ldr x20, [sp, 0x200]
            ; ldr x21, [sp, 0x208]
            ; ldr x22, [sp, 0x210]
            ; ldr x23, [sp, 0x218]
            ; ldr x24, [sp, 0x220]

            // Fix up the stack
            ; add sp, sp, self.0.mem_offset as u32
            ; ret
        );

        self.0.ops.finalize()
    }
}

impl FloatSliceAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "C" fn(f32) -> f32,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2
            ; mov x23, x3

            // We use registers v8-v15 (callee saved, but only lower 64 bytes)
            // and v16-v31 (caller saved)
            // TODO: track which registers are actually used?
            ; stp q8, q9, [sp, 0x50]
            ; stp q10, q11, [sp, 0x70]
            ; stp q12, q13, [sp, 0x90]
            ; stp q14, q15, [sp, 0xb0]
            ; stp q16, q17, [sp, 0xd0]
            ; stp q18, q19, [sp, 0xf0]
            ; stp q20, q21, [sp, 0x110]
            ; stp q22, q23, [sp, 0x130]
            ; stp q24, q25, [sp, 0x150]
            ; stp q26, q27, [sp, 0x170]
            ; stp q28, q29, [sp, 0x190]
            ; stp q30, q31, [sp, 0x1b0]

            // Load the function address, awkwardly, into a callee-saved
            // register (so we only need to do this once)
            ; movz x24, ((addr >> 48) as u32), lsl 48
            ; movk x24, ((addr >> 32) as u32), lsl 32
            ; movk x24, ((addr >> 16) as u32), lsl 16
            ; movk x24, addr as u32

            // We're going to back up our argument into d8/d9 (since the callee
            // only saves the bottom 64 bits).  Note that d8/d9 may be our input
            // argument, so we'll move it to v0 first.
            ; mov v0.b16, V(reg(arg_reg)).b16
            ; mov d8, v0.d[0]
            ; mov d9, v0.d[1]

            ; mov s0, v8.s[0]
            ; blr x24
            ; mov v8.s[0], v0.s[0]

            ; mov s0, v8.s[1]
            ; blr x24
            ; mov v8.s[1], v0.s[0]

            ; mov s0, v9.s[0]
            ; blr x24
            ; mov v9.s[0], v0.s[0]

            ; mov s0, v9.s[1]
            ; blr x24
            ; mov v9.s[1], v0.s[0]

            // Copy into v0, because we're about to restore v8
            ; mov v0.d[0], v8.d[0]
            ; mov v0.d[1], v9.d[0]

            // Restore register state
            ; ldp q8, q9, [sp, 0x50]
            ; ldp q10, q11, [sp, 0x70]
            ; ldp q12, q13, [sp, 0x90]
            ; ldp q14, q15, [sp, 0xb0]
            ; ldp q16, q17, [sp, 0xd0]
            ; ldp q18, q19, [sp, 0xf0]
            ; ldp q20, q21, [sp, 0x110]
            ; ldp q22, q23, [sp, 0x130]
            ; ldp q24, q25, [sp, 0x150]
            ; ldp q26, q27, [sp, 0x170]
            ; ldp q28, q29, [sp, 0x190]
            ; ldp q30, q31, [sp, 0x1b0]

            // Set our output value
            ; mov V(reg(out_reg)).b16, v0.b16

            // Restore our current state
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x22
            ; mov x3, x23
        );
    }
    fn call_fn_binary(
        &mut self,
        out_reg: u8,
        lhs_reg: u8,
        rhs_reg: u8,
        f: extern "C" fn(f32, f32) -> f32,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2
            ; mov x23, x3

            // We use registers v8-v15 (callee saved, but only lower 64 bytes)
            // and v16-v31 (caller saved)
            // TODO: track which registers are actually used?
            ; stp q8, q9, [sp, 0x50]
            ; stp q10, q11, [sp, 0x70]
            ; stp q12, q13, [sp, 0x90]
            ; stp q14, q15, [sp, 0xb0]
            ; stp q16, q17, [sp, 0xd0]
            ; stp q18, q19, [sp, 0xf0]
            ; stp q20, q21, [sp, 0x110]
            ; stp q22, q23, [sp, 0x130]
            ; stp q24, q25, [sp, 0x150]
            ; stp q26, q27, [sp, 0x170]
            ; stp q28, q29, [sp, 0x190]
            ; stp q30, q31, [sp, 0x1b0]

            // Load the function address, awkwardly, into a callee-saved
            // register (so we only need to do this once)
            ; movz x24, ((addr >> 48) as u32), lsl 48
            ; movk x24, ((addr >> 32) as u32), lsl 32
            ; movk x24, ((addr >> 16) as u32), lsl 16
            ; movk x24, addr as u32

            // We're going to back up our argument into d8/d9/d10/d11 (since the
            // callee only saves the bottom 64 bits).  Note that d8/d9/d10/d11
            // may be our input argument, so we'll move it to v0/v1 first.
            ; mov v0.b16, V(reg(lhs_reg)).b16
            ; mov v1.b16, V(reg(rhs_reg)).b16
            ; mov d8, v0.d[0]
            ; mov d9, v0.d[1]
            ; mov d10, v1.d[0]
            ; mov d11, v1.d[1]

            ; mov s0, v8.s[0]
            ; mov s1, v10.s[0]
            ; blr x24
            ; mov v8.s[0], v0.s[0]

            ; mov s0, v8.s[1]
            ; mov s1, v10.s[1]
            ; blr x24
            ; mov v8.s[1], v0.s[0]

            ; mov s0, v9.s[0]
            ; mov s1, v11.s[0]
            ; blr x24
            ; mov v9.s[0], v0.s[0]

            ; mov s0, v9.s[1]
            ; mov s1, v11.s[1]
            ; blr x24
            ; mov v9.s[1], v0.s[0]

            // Copy into v0, because we're about to restore v8
            ; mov v0.d[0], v8.d[0]
            ; mov v0.d[1], v9.d[0]

            // Restore register state
            ; ldp q8, q9, [sp, 0x50]
            ; ldp q10, q11, [sp, 0x70]
            ; ldp q12, q13, [sp, 0x90]
            ; ldp q14, q15, [sp, 0xb0]
            ; ldp q16, q17, [sp, 0xd0]
            ; ldp q18, q19, [sp, 0xf0]
            ; ldp q20, q21, [sp, 0x110]
            ; ldp q22, q23, [sp, 0x130]
            ; ldp q24, q25, [sp, 0x150]
            ; ldp q26, q27, [sp, 0x170]
            ; ldp q28, q29, [sp, 0x190]
            ; ldp q30, q31, [sp, 0x1b0]

            // Set our output value
            ; mov V(reg(out_reg)).b16, v0.b16

            // Restore our current state
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x22
            ; mov x3, x23
        );
    }
}
