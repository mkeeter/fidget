use crate::jit::{
    float_slice::FloatSliceAssembler, mmap::Mmap, reg, Assembler,
    AssemblerData, Error, IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub const SIMD_WIDTH: usize = 4;

/// Assembler for SIMD point-wise evaluation on `aarch64`
///
/// | Argument | Register | Type                |
/// | ---------|----------|---------------------|
/// | X        | `x0`     | `*const [f32; 4]`   |
/// | Y        | `x1`     | `*const [f32; 4]`   |
/// | Z        | `x2`     | `*const [f32; 4]`   |
/// | vars     | `x3`     | `*const f32`        |
/// | out      | `x4`     | `*mut [f32; 4]`     |
/// | size     | `x5`     | `u64`               |
///
/// The arrays (other than `vars`) must be an even multiple of 4 floats, since
/// we're using NEON and 128-bit wide operations for everything.  The `vars`
/// array contains single `f32` values, which are broadcast into SIMD registers
/// when they are used.
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`
impl Assembler for FloatSliceAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
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
        out.prepare_stack(slot_count);

        dynasm!(out.ops
            // The loop returns here, and we check whether we need to loop
            ; ->L:
            // Remember, at this point we have
            //  x0: x input array pointer
            //  x1: y input array pointer
            //  x2: z input array pointer
            //  x3: vars input array pointer (non-advancing)
            //  x4: output array pointer
            //  x5: number of points to evaluate
            //
            // We'll be advancing x0, x1, x2 here (and decrementing x5 by 4);
            // x4 is advanced in finalize().

            ; cmp x5, #0
            ; b.ne #32 // skip to loop body

            // fini:
            // This is our finalization code, which happens after all evaluation
            // is complete.
            //
            // Restore stack space used for spills
            ; add   sp, sp, #(out.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret

            // Loop body:
            //
            // Load V0/1/2.S4 with X/Y/Z values, post-increment
            //
            // We're actually loading two f32s, but we can pretend they're
            // doubles in order to move 64 bits at a time
            ; ldr q0, [x0], #16
            ; ldr q1, [x1], #16
            ; ldr q2, [x2], #16
            ; sub x5, x5, #4 // We handle 4 items at a time
        );

        Self(out)
    }

    fn bytes_per_clause() -> usize {
        10
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; ldr Q(reg(dst_reg)), [sp, #(sp_offset)]
        )
    }

    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; str Q(reg(src_reg)), [sp, #(sp_offset)]
        )
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(src_arg as u32).b16);
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0.ops
            ; ldr w15, [x3, #(src_arg * 4)]
            ; dup V(reg(out_reg)).s4, w15
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_sin(f: f32) -> f32 {
            f.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_sin);
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

    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x4
            // It's fine to overwrite X at this point in V0, since we're not
            // using it anymore.
            ; mov v0.d[0], V(reg(out_reg)).d[1]
            ; stp D(reg(out_reg)), d0, [x4], #16
            ; b ->L
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
            // Back up our current state to caller-saved registers
            ; mov x10, x0
            ; mov x11, x1
            ; mov x12, x2
            ; mov x13, x3
            ; mov x14, x4
            ; mov x15, x5

            // Back up X/Y/Z values
            ; stp q0, q1, [sp, #-32]!
            ; stp q2, q3, [sp, #-32]!

            // We use registers v8-v15 (callee saved, but only lower 64 bytes)
            // and v16-v31 (caller saved)
            ; stp q8, q9, [sp, #-32]!
            ; stp q10, q11, [sp, #-32]!
            ; stp q12, q13, [sp, #-32]!
            ; stp q14, q15, [sp, #-32]!
            ; stp q16, q17, [sp, #-32]!
            ; stp q18, q19, [sp, #-32]!
            ; stp q20, q21, [sp, #-32]!
            ; stp q22, q23, [sp, #-32]!
            ; stp q24, q25, [sp, #-32]!
            ; stp q26, q27, [sp, #-32]!
            ; stp q28, q29, [sp, #-32]!
            ; stp q30, q31, [sp, #-32]!

            // Load the function address, awkwardly, into a caller-saved
            // register (so we only need to do this once)
            ; movz x9, #((addr >> 48) as u32), lsl 48
            ; movk x9, #((addr >> 32) as u32), lsl 32
            ; movk x9, #((addr >> 16) as u32), lsl 16
            ; movk x9, #(addr as u32)

            // We're going to back up our argument into d8/d9 (since the callee
            // only saves the bottom 64 bits).  Note that d8/d9 may be our input
            // argument, so we'll move it to v4 first.
            ; mov v4.b16, V(reg(arg_reg)).b16
            ; mov d8, v4.d[0]
            ; mov d9, v4.d[1]

            ; mov s0, v8.s[0]
            ; blr x9
            ; mov v8.s[0], v0.s[0]

            ; mov s0, v8.s[1]
            ; blr x9
            ; mov v8.s[1], v0.s[0]

            ; mov s0, v9.s[0]
            ; blr x9
            ; mov v9.s[0], v0.s[0]

            ; mov s0, v9.s[1]
            ; blr x9
            ; mov v9.s[1], v0.s[0]

            // Copy into v4, because we're about to restore v8
            ; mov v4.s[0], v8.s[0]
            ; mov v4.s[1], v8.s[1]
            ; mov v4.s[2], v9.s[0]
            ; mov v4.s[3], v9.s[1]

            // Restore register state (lol)
            ; ldp q30, q31, [sp], #32
            ; ldp q28, q29, [sp], #32
            ; ldp q26, q27, [sp], #32
            ; ldp q24, q25, [sp], #32
            ; ldp q22, q23, [sp], #32
            ; ldp q20, q21, [sp], #32
            ; ldp q18, q19, [sp], #32
            ; ldp q16, q17, [sp], #32
            ; ldp q14, q15, [sp], #32
            ; ldp q12, q13, [sp], #32
            ; ldp q10, q11, [sp], #32
            ; ldp q8, q9, [sp], #32

            ; ldp q2, q3, [sp], #32
            ; ldp q0, q1, [sp], #32

            ; mov x0, x10
            ; mov x1, x11
            ; mov x2, x12
            ; mov x3, x13
            ; mov x4, x14
            ; mov x5, x15

            // Set our output value
            ; mov V(reg(out_reg)).b16, v4.b16
        );
    }
}
