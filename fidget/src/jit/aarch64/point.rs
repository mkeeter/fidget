use crate::{
    Error,
    jit::{
        Assembler, AssemblerData, CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT,
        IMM_REG, OFFSET, REGISTER_LIMIT, mmap::Mmap, point::PointAssembler,
        reg,
    },
};
use dynasmrt::{DynasmApi, dynasm};

/// Implementation for the single-point assembler on `aarch64`
///
/// Registers are passed in as follows:
///
/// | Variable   | Register | Type                  |
/// |------------|----------|-----------------------|
/// | `vars`     | `x0`     | `*const f32` (array)  |
/// | `choices`  | `x1`     | `*mut u8` (array)     |
/// | `simplify` | `x2`     | `*mut u8` (single)    |
/// | `output`   | `x3`     | `*mut f32` (array)    |
///
/// During evaluation, registers are identical.  In addition, we use the
/// following registers during evaluation:
///
/// | Register | Description                                          |
/// |----------|------------------------------------------------------|
/// | `s3`     | Immediate value (`IMM_REG`)                          |
/// | `s7`     | Immediate value for `recip` (1.0)                    |
/// | `s8-15`  | Tape values (callee-saved)                           |
/// | `s16-31` | Tape values (caller-saved)                           |
/// | `x0`     | Function pointer for calls                           |
/// | `w9`     | Staging for loading immediate                        |
/// | `w14`    | Choice byte (limited scope)                          |
/// | `x20`    | Backup for `x0` during function calls (callee-saved) |
/// | `x21`    | Backup for `x1` during function calls (callee-saved) |
/// | `x22`    | Backup for `x2` during function calls (callee-saved) |
/// | `x23`    | Backup for `x3` during function calls (callee-saved) |
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0xa8     | `x23`        | During functions calls, we use these        |
/// | 0xa0     | `x22`        | as temporary storage so must preserve their |
/// | 0x98     | `x21`        | previous values on the stack                |
/// | 0x90     | `x20`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      |              | Alignment padding                           |
/// |----------|--------------|---------------------------------------------|
/// | 0x8c     | `s31`        | During functions calls, caller-saved tape   |
/// | 0x88     | `s30`        | registers are saved on the stack            |
/// | 0x84     | `s29`        |                                             |
/// | 0x80     | `s28`        |                                             |
/// | 0x7c     | `s27`        |                                             |
/// | 0x78     | `s26`        |                                             |
/// | 0x74     | `s25`        |                                             |
/// | 0x70     | `s24`        |                                             |
/// | 0x6c     | `s23`        |                                             |
/// | 0x68     | `s22`        |                                             |
/// | 0x64     | `s21`        |                                             |
/// | 0x60     | `s20`        |                                             |
/// | 0x5c     | `s19`        |                                             |
/// | 0x58     | `s18`        |                                             |
/// | 0x54     | `s17`        |                                             |
/// | 0x50     | `s16`        |                                             |
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
const STACK_SIZE: u32 = 0xb0;

#[allow(clippy::unnecessary_cast)] // dynasm-rs#106
impl Assembler for PointAssembler {
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
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; ldr S(reg(dst_reg)), [sp, sp_offset])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem) + STACK_SIZE;
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; str S(reg(src_reg)), [sp, sp_offset])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg < 16384 / 4);
        dynasm!(self.0.ops
            ; ldr S(reg(out_reg)), [x0, src_arg * 4]
        );
    }
    /// Copies the register to the output
    fn build_output(&mut self, arg_reg: u8, output_index: u32) {
        assert!(output_index < 16384 / 4);
        dynasm!(self.0.ops
            ; str S(reg(arg_reg)), [x3, output_index * 4]
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmov S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn point_sin(v: f32) -> f32 {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, point_sin);
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
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fneg S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fabs S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov s7, 1.0
            ; fdiv S(reg(out_reg)), s7, S(reg(lhs_reg))
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fsqrt S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(lhs_reg)))
    }

    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; mvn v6.b8, v6.b8

            // Round, then convert back to f32
            ; fcvtms S(reg(out_reg)), S(reg(lhs_reg))
            ; scvtf S(reg(out_reg)), S(reg(out_reg))

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B8, V(reg(out_reg)).B8, v6.b8
        );
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; mvn v6.b8, v6.b8

            // Round, then convert back to f32
            ; fcvtps S(reg(out_reg)), S(reg(lhs_reg))
            ; scvtf S(reg(out_reg)), S(reg(out_reg))

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B8, V(reg(out_reg)).B8, v6.b8
        );
    }
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a NAN mask
            ; fcmeq s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; mvn v6.b8, v6.b8

            // Round, then convert back to f32
            ; fcvtas S(reg(out_reg)), S(reg(lhs_reg))
            ; scvtf S(reg(out_reg)), S(reg(out_reg))

            // Apply the NAN mask
            ; orr V(reg(out_reg)).B8, V(reg(out_reg)).B8, v6.b8
        );
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
    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "C" fn float_atan2(y: f32, x: f32) -> f32 {
            y.atan2(x)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, float_atan2);
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; ldrb w14, [x1]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi 20 // -> RHS
            ; b.gt 32 // -> LHS

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmax S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, CHOICE_BOTH
            ; b 32 // -> end

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, CHOICE_RIGHT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 16

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, CHOICE_LEFT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            // fall-through to end

            // <- end
            ; strb w14, [x1], 1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; ldrb w14, [x1]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi 20
            ; b.gt 32

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmin S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, CHOICE_BOTH
            ; b 32 // -> end

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, CHOICE_LEFT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 16

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, CHOICE_RIGHT
            ; strb w14, [x2, 0]
            // fall-through to end

            // <- end
            ; strb w14, [x1], 1 // post-increment
        )
    }

    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fabs s6, S(reg(rhs_reg))
            ; fdiv s7, S(reg(lhs_reg)), s6
            ; frintm s7, s7 // round down
            ; fmul s7, s7, s6
            ; fsub S(reg(out_reg)), S(reg(lhs_reg)), s7
        )
    }

    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(arg_reg)), 0.0
            ; fmov S(reg(out_reg)), 1.0
            ; and V(reg(out_reg)).b8, V(reg(out_reg)).b8, v6.b8
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(lhs_reg)), 0.0
            ; fmov w10, s6 // s6 = w10 = (lhs == 0)
            ; mov w9, CHOICE_LEFT
            ; and w9, w9, w10 // w9 = (lhs == 0) ? CHOICE_LEFT : 0

            ; mvn w10, w10
            ; mov w11, CHOICE_RIGHT
            ; and w11, w11, w10 // w11 = (lhs != 0) ? CHOICE_RIGHT : 0
            ; orr w11, w11, w9  // w11 = choice to write

            ; ldrb w14, [x1]
            ; orr w14, w14, w11
            ; strb w14, [x1], 1 // post-increment

            ; strb w14, [x2, 0] // store any non-zero value to `simplify`

            // Accumulate our output value
            ; and v5.b8, v6.b8, V(reg(lhs_reg)).b8
            ; mvn v6.b8, v6.b8
            ; and v6.b8, v6.b8, V(reg(rhs_reg)).b8
            ; orr V(reg(out_reg)).b8, v5.b8, v6.b8
        );
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(lhs_reg)), 0.0
            ; mvn v6.b8, v6.b8

            ; fmov w10, s6 // s6 = w10 = (lhs != 0)
            ; mov w9, CHOICE_LEFT
            ; and w9, w9, w10 // w9 = (lhs != 0) ? CHOICE_LEFT : 0

            ; mvn w10, w10
            ; mov w11, CHOICE_RIGHT
            ; and w11, w11, w10 // w11 = (lhs == 0) ? CHOICE_RIGHT : 0
            ; orr w11, w11, w9  // w11 = choice to write

            ; ldrb w14, [x1]
            ; orr w14, w14, w11
            ; strb w14, [x1], 1 // post-increment

            ; strb w14, [x2, 0] // store any non-zero value to `simplify`

            // Accumulate our output value
            ; and v5.b8, v6.b8, V(reg(lhs_reg)).b8
            ; mvn v6.b8, v6.b8
            ; and v6.b8, v6.b8, V(reg(rhs_reg)).b8
            ; orr V(reg(out_reg)).b8, v5.b8, v6.b8
        );
    }

    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // This is using SIMD instructions to avoid branch; dunno if it's faster
        // but it means we can use very similar code to float / grad slice
        // evaluators.
        dynasm!(self.0.ops
            // Build a mask of NAN positions in s6
            ; fcmeq s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; fcmeq s7, S(reg(rhs_reg)), S(reg(rhs_reg))
            ; and v6.b8, v6.b8, v7.b8
            ; mvn v6.b8, v6.b8

            // Build our two comparisons
            ; fcmgt s4, S(reg(rhs_reg)), S(reg(lhs_reg))
            ; fcmgt s5, S(reg(lhs_reg)), S(reg(rhs_reg))

            // Apply -1 value (if relevant)
            ; fmov s7, -1.0
            ; and V(reg(out_reg)).B8, v4.B8, v7.B8

            // Apply +1 value (if relevant)
            ; fmov s7, 1.0
            ; and v5.B8, v5.B8, v7.B8
            ; orr V(reg(out_reg)).B8, V(reg(out_reg)).B8, v5.B8

            // Apply NAN value
            ; mov w9, f32::NAN.to_bits()
            ; fmov s7, w9
            ; and v7.b8, v7.b8, v6.b8

            // Apply NAN mask to NAN positions
            ; orr V(reg(out_reg)).B8, V(reg(out_reg)).B8, v7.b8
        );
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        if imm_u32 & 0xFFFF == 0 {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 >> 16, lsl 16
                ; fmov S(IMM_REG as u32), w9
            );
        } else if imm_u32 & 0xFFFF_0000 == 0 {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 & 0xFFFF
                ; fmov S(IMM_REG as u32), w9
            );
        } else {
            dynasm!(self.0.ops
                ; movz w9, imm_u32 >> 16, lsl 16
                ; movk w9, imm_u32 & 0xFFFF
                ; fmov S(IMM_REG as u32), w9
            );
        }
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self) -> Result<Mmap, Error> {
        if self.0.saved_callee_regs {
            dynasm!(self.0.ops
                // Restore callee-saved registers
                ; ldp x20, x21, [sp, 0x90]
                ; ldr x22, [sp, 0xa0]
                ; ldr x23, [sp, 0xa8]
            )
        }
        dynasm!(self.0.ops
            // Restore frame and link register
            ; ldp   x29, x30, [sp, 0x0]

            // Restore callee-saved floating-point registers
            ; ldp   d8, d9, [sp, 0x10]
            ; ldp   d10, d11, [sp, 0x20]
            ; ldp   d12, d13, [sp, 0x30]
            ; ldp   d14, d15, [sp, 0x40]
        );
        self.0.finalize()
    }
}

impl PointAssembler {
    fn ensure_callee_regs_saved(&mut self) {
        if !self.0.saved_callee_regs {
            dynasm!(self.0.ops
                // Back up a few callee-saved registers that we're about to use
                ; stp x20, x21, [sp, 0x90]
                ; str x22, [sp, 0xa0]
                ; str x23, [sp, 0xa8]
            );
            self.0.saved_callee_regs = true;
        }
    }

    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "C" fn(f32) -> f32,
    ) {
        self.ensure_callee_regs_saved();
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state to callee-saved registers
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2
            ; mov x23, x3

            // Back up our state
            ; stp s16, s17, [sp, 0x50]
            ; stp s18, s19, [sp, 0x58]
            ; stp s20, s21, [sp, 0x60]
            ; stp s22, s23, [sp, 0x68]
            ; stp s24, s25, [sp, 0x70]
            ; stp s26, s27, [sp, 0x78]
            ; stp s28, s29, [sp, 0x80]
            ; stp s30, s31, [sp, 0x88]

            // Load the function address, awkwardly, into x0 (it doesn't matter
            // that it's about to be overwritten, because we only call it once)
            ; movz x0, (addr >> 48) as u32 & 0xFFFF, lsl 48
            ; movk x0, (addr >> 32) as u32 & 0xFFFF, lsl 32
            ; movk x0, (addr >> 16) as u32 & 0xFFFF, lsl 16
            ; movk x0, addr as u32 & 0xFFFF

            ; fmov s0, S(reg(arg_reg))
            ; blr x0

            // Restore floating-point state
            ; ldp s16, s17, [sp, 0x50]
            ; ldp s18, s19, [sp, 0x58]
            ; ldp s20, s21, [sp, 0x60]
            ; ldp s22, s23, [sp, 0x68]
            ; ldp s24, s25, [sp, 0x70]
            ; ldp s26, s27, [sp, 0x78]
            ; ldp s28, s29, [sp, 0x80]
            ; ldp s30, s31, [sp, 0x88]

            // Set our output value
            ; fmov S(reg(out_reg)), s0

            // Restore X/Y/Z values
            ; ldp s0, s1, [sp, 0x90]
            ; ldr s2, [sp, 0x98]

            // Restore registers
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
        self.ensure_callee_regs_saved();
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state to callee-saved registers
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2

            // Back up our state
            ; stp s16, s17, [sp, 0x50]
            ; stp s18, s19, [sp, 0x58]
            ; stp s20, s21, [sp, 0x60]
            ; stp s22, s23, [sp, 0x68]
            ; stp s24, s25, [sp, 0x70]
            ; stp s26, s27, [sp, 0x78]
            ; stp s28, s29, [sp, 0x80]
            ; stp s30, s31, [sp, 0x88]

            // Load the function address, awkwardly, into x0 (it doesn't matter
            // that it's about to be overwritten, because we only call it once)
            ; movz x0, (addr >> 48) as u32 & 0xFFFF, lsl 48
            ; movk x0, (addr >> 32) as u32 & 0xFFFF, lsl 32
            ; movk x0, (addr >> 16) as u32 & 0xFFFF, lsl 16
            ; movk x0, addr as u32 & 0xFFFF

            ; fmov s0, S(reg(lhs_reg))
            ; fmov s1, S(reg(rhs_reg))
            ; blr x0

            // Restore floating-point state
            ; ldp s16, s17, [sp, 0x50]
            ; ldp s18, s19, [sp, 0x58]
            ; ldp s20, s21, [sp, 0x60]
            ; ldp s22, s23, [sp, 0x68]
            ; ldp s24, s25, [sp, 0x70]
            ; ldp s26, s27, [sp, 0x78]
            ; ldp s28, s29, [sp, 0x80]
            ; ldp s30, s31, [sp, 0x88]

            // Set our output value
            ; fmov S(reg(out_reg)), s0

            // Restore X/Y/Z values
            ; ldp s0, s1, [sp, 0x90]
            ; ldr s2, [sp, 0x98]

            // Restore registers
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x22
        );
    }
}
