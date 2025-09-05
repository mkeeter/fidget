use crate::{
    Error,
    jit::{
        Assembler, AssemblerData, IMM_REG, OFFSET, REGISTER_LIMIT,
        grad_slice::GradSliceAssembler, mmap::Mmap, reg,
    },
    types::Grad,
};
use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

/// Implementation for the gradient slice assembler on `aarch64`
///
/// Registers are passed in as follows:
///
/// | Variable   | Register | Type                      |
/// |------------|----------|---------------------------|
/// | `vars`     | `x0`     | `*const *const [f32; 4]`  |
/// | `out`      | `x1`     | `*const *mut [f32; 4]`    |
/// | `count`    | `x2`     | `u64`                     |
///
/// During evaluation, variables are loaded into SIMD registers in the order
/// `[value, dx, dy, dz]`, e.g. if we load `vars[0]` into `V0`, its value would
/// be in `V0.S0` (and the three partial derivatives would be in `V0.S{1,2,3}`).
///
/// In addition to the registers above (`x0-5`), the following extra registers
/// are used during evaluation:
///
/// | Register | Description                                          |
/// |----------|------------------------------------------------------|
/// | `x3`     | byte offset within input arrays                      |
/// | `x4`     | Staging for loading SIMD values                      |
/// | `v3.s4`  | Immediate value (`IMM_REG`)                          |
/// | `v7.s4`  | Immediate value for recip (1.0)                      |
/// | `w9`     | Staging for loading immediates                       |
/// | `w15`    | Staging to load variables                            |
/// | `x20-23` | Backups for `x0-3` during function calls             |
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0x220    | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
/// | 0x218    | `x23`        | Backup for callee-saved registers           |
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
const STACK_SIZE: u32 = 0x220;

#[allow(clippy::unnecessary_cast)] // dynasm-rs#106
impl Assembler for GradSliceAssembler {
    type Data = Grad;

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

            ; mov x3, 0

            // The loop returns here, and we check whether we need to loop
            ; ->L:
            // Remember, at this point we have
            //  x0: vars input pointer
            //  x1: output array pointer
            //  x2: number of points to evaluate
            //  x3: offset within SIMD arrays
            //
            // We'll be decrementing x2 by 1 here; x1 and x3 are modified in
            // finalize().

            ; cmp x2, 0
            ; b.eq ->E // function exit

            // Loop body: math begins below
        );

        Self(out)
    }

    fn bytes_per_clause() -> usize {
        20
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
            ; eor V(reg(out_reg)).b16, V(reg(out_reg)).b16, V(reg(out_reg)).b16
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
        extern "C" fn grad_sin(v: Grad) -> Grad {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, grad_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_cos(f: Grad) -> Grad {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_tan(f: Grad) -> Grad {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_asin(f: Grad) -> Grad {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_acos(f: Grad) -> Grad {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_atan(f: Grad) -> Grad {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_exp(f: Grad) -> Grad {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_ln(f: Grad) -> Grad {
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
        // TODO: use two fcsel instead?
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), 0.0
            ; b.lt 12 // -> neg
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; b 8 // -> end
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
            ; fmov s6, 1.0
            ; fdiv s6, s6, S(reg(lhs_reg))
            ; mov V(reg(out_reg)).b16, v7.b16
            ; mov V(reg(out_reg)).s[0], v6.s[0]
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fsqrt s6, S(reg(lhs_reg))
            ; fmov s7, 2.0
            ; fmul s7, s6, s7
            ; dup v7.s4, v7.s[0]
            ; fdiv V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, v7.s4
            ; mov V(reg(out_reg)).S[0], v6.S[0]
        )
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmov s7, 2.0
            ; dup v7.s4, v7.s[0]
            ; fmov s6, 1.0
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

    // TODO hand-write these functions
    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn grad_floor(v: Grad) -> Grad {
            v.floor()
        }
        self.call_fn_unary(out_reg, lhs_reg, grad_floor);
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn grad_ceil(v: Grad) -> Grad {
            v.ceil()
        }
        self.call_fn_unary(out_reg, lhs_reg, grad_ceil);
    }
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn grad_round(v: Grad) -> Grad {
            v.round()
        }
        self.call_fn_unary(out_reg, lhs_reg, grad_round);
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
            ; fmls v5.s4, v6.s4, V(reg(rhs_reg)).s4
            // At this point, gradients are of the form
            //      rhs.v * lhs.d - lhs.v * rhs.d

            // Divide by rhs.v**2
            ; fmul s6, S(reg(rhs_reg)), S(reg(rhs_reg))
            ; fmov w9, s6
            ; dup v6.s4, w9
            ; fdiv v5.s4, v5.s4, v6.s4

            // Patch in the actual division value
            ; fdiv s6, S(reg(lhs_reg)), S(reg(rhs_reg))
            ; mov V(reg(out_reg)).b16, v5.b16
            ; mov V(reg(out_reg)).s[0], v6.s[0]
        )
    }

    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "C" fn grad_atan2(y: Grad, x: Grad) -> Grad {
            y.atan2(x)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, grad_atan2);
    }

    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.vs 24 // -> NaN
            ; b.gt 12 // -> lhs

            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b 20 // -> end

            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; b 12 // -> end

            // NaN handler
            ; mov w9, f32::NAN.to_bits()
            ; fmov  S(reg(out_reg)), w9
            // end:
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.vs 24 // -> NaN
            ; b.lt 12 // -> lhs

            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b 20 // -> end

            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; b 12 // -> end

            // NaN handler
            ; mov w9, f32::NAN.to_bits()
            ; fmov  S(reg(out_reg)), w9
            // end:
        )
    }

    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "C" fn grad_modulo(lhs: Grad, rhs: Grad) -> Grad {
            lhs.rem_euclid(rhs)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, grad_modulo);
    }

    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(arg_reg)), 0.0
            ; fmov S(reg(out_reg)), 1.0
            ; and V(reg(out_reg)).b16, V(reg(out_reg)).b16, v6.b16
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(lhs_reg)), 0.0
            ; dup v6.s4, v6.s[0]
            ; mvn v7.b16, v6.b16
            ; and v6.b16, v6.b16, V(reg(lhs_reg)).b16
            ; and v7.b16, v7.b16, V(reg(rhs_reg)).b16
            ; orr V(reg(out_reg)).b16, v6.b16, v7.b16
        );
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmeq s6, S(reg(lhs_reg)), 0.0
            ; dup v6.s4, v6.s[0]
            ; mvn v7.b16, v6.b16
            ; and v7.b16, v7.b16, V(reg(lhs_reg)).b16
            ; and v6.b16, v6.b16, V(reg(rhs_reg)).b16
            ; orr V(reg(out_reg)).b16, v6.b16, v7.b16
        );
    }

    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Check whether either argument is NAN
            ; fcmeq s6, S(reg(lhs_reg)), S(reg(lhs_reg))
            ; dup v6.s4, v6.s[0]
            ; fcmeq s7, S(reg(rhs_reg)), S(reg(rhs_reg))
            ; dup v7.s4, v7.s[0]
            ; and v6.b16, v6.b16, v7.b16
            ; mvn v6.b16, v6.b16
            // At this point, v6 is all 1s if either argument is NAN

            // build masks for all 1s / all 0s
            ; fcmgt s4, S(reg(rhs_reg)), S(reg(lhs_reg))
            ; dup v4.s4, v4.s[0]
            ; fcmgt s5, S(reg(lhs_reg)), S(reg(rhs_reg))
            ; dup v5.s4, v5.s[0]

            // (lhs < rhs) & [1.0, 0.0, 0.0, 0.0]
            ; fmov s7, -1.0
            ; and V(reg(out_reg)).b16, v4.b16, v7.b16

            ; fmov s7, 1.0
            ; and v5.B16, v5.B16, v7.B16
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v5.B16

            // Build NAN mask
            ; mov w9, f32::NAN.to_bits()
            ; fmov s7, w9
            ; and v7.b16, v7.b16, v6.b16

            // Build NAN to output
            ; orr V(reg(out_reg)).B16, V(reg(out_reg)).B16, v7.b16
        )
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
        dynasm!(self.0.ops
            // update our "items remaining" counter
            ; sub x2, x2, 1 // We handle 1 item at a time

            // Adjust the array offset pointer
            ; add x3, x3, 16 // 1 item = 16 bytes

            ; b ->L // Jump back to the loop start

            ; ->E:
            // This is our finalization code, which happens after all evaluation
            // is complete.
            //
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
        );
        self.0.finalize()
    }
}

impl GradSliceAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "C" fn(Grad) -> Grad,
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

            // Load the function address, awkwardly, into x0 (it doesn't matter
            // that it can be overwritten, because we're only ever calling it
            // once)
            ; movz x0, (addr >> 48) as u32 & 0xFFFF, lsl 48
            ; movk x0, (addr >> 32) as u32 & 0xFFFF, lsl 32
            ; movk x0, (addr >> 16) as u32 & 0xFFFF, lsl 16
            ; movk x0, addr as u32 & 0xFFFF

            // Prepare to call our stuff!
            ; mov s0, V(reg(arg_reg)).s[0]
            ; mov s1, V(reg(arg_reg)).s[1]
            ; mov s2, V(reg(arg_reg)).s[2]
            ; mov s3, V(reg(arg_reg)).s[3]

            ; blr x0

            // Restore register state (lol)
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

            ; mov V(reg(out_reg)).s[0], v0.s[0]
            ; mov V(reg(out_reg)).s[1], v1.s[0]
            ; mov V(reg(out_reg)).s[2], v2.s[0]
            ; mov V(reg(out_reg)).s[3], v3.s[0]

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
        f: extern "C" fn(Grad, Grad) -> Grad,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2
            ; mov x23, x3

            // Back up X/Y/Z values (TODO use registers here as well?)
            ; stp q0, q1, [sp, 0x1d0]
            ; str q2, [sp, 0x1f0]

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

            // Load the function address, awkwardly, into x0 (it doesn't matter
            // that it could be thrashed by the call, since we're only calling
            // it once).
            ; movz x0, (addr >> 48) as u32 & 0xFFFF, lsl 48
            ; movk x0, (addr >> 32) as u32 & 0xFFFF, lsl 32
            ; movk x0, (addr >> 16) as u32 & 0xFFFF, lsl 16
            ; movk x0, addr as u32 & 0xFFFF

            // Prepare to call our stuff!
            ; mov s0, V(reg(lhs_reg)).s[0]
            ; mov s1, V(reg(lhs_reg)).s[1]
            ; mov s2, V(reg(lhs_reg)).s[2]
            ; mov s4, V(reg(rhs_reg)).s[0]
            ; mov s5, V(reg(rhs_reg)).s[1]
            ; mov s6, V(reg(rhs_reg)).s[2]
            ; mov s7, V(reg(rhs_reg)).s[3]

            // We do s3 last because it could be IMM_REG
            ; mov s3, V(reg(lhs_reg)).s[3]

            ; blr x0

            // Restore register state (lol)
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

            // Copy into the out register before restoring v0 (aka q0)
            ; mov V(reg(out_reg)).s[0], v0.s[0]
            ; mov V(reg(out_reg)).s[1], v1.s[0]
            ; mov V(reg(out_reg)).s[2], v2.s[0]
            ; mov V(reg(out_reg)).s[3], v3.s[0]

            // Restore X/Y/Z values
            ; ldp q0, q1, [sp, 0x1d0]
            ; ldr q2, [sp, 0x1f0]

            // Restore our current state
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x22
            ; mov x3, x23
        );
    }
}
