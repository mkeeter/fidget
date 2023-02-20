use crate::{
    eval::types::Grad,
    jit::{
        mmap::Mmap, reg, AssemblerData, AssemblerT, JitBulkEval, SimdAssembler,
        IMM_REG, OFFSET, REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi};

/// Assembler for automatic differentiation / gradient evaluation
///
///
/// Arguments are passed as 3x `*const f32` in `x0-2`, a var array in
/// `x3`, and an output array `*mut f32` in `x4`.  Each pointer in the input
/// arrays represents 4x `f32`, the var array is single `f32`s, and the `out`
/// array is `[v, dx, dy, dz]`
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`.  Each SIMD register
/// is in the order `[value, dx, dy, dz]`, e.g. the value for X is in `V0.S0`.
pub struct GradSliceAssembler(AssemblerData<[f32; 4]>, usize);

#[cfg(target_arch = "aarch64")]
impl AssemblerT for GradSliceAssembler {
    type Data = Grad;

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
        let loop_start = out.ops.len();

        dynasm!(out.ops
            // The loop returns here, and we check whether we need to loop
            // Remember, at this point we have
            //  x0: x input array pointer
            //  x1: y input array pointer
            //  x2: z input array pointer
            //  x3: vars input array pointer (non-advancing)
            //  x4: output array pointer
            //  x5: number of points to evaluate
            //
            // We'll be advancing x0, x1, x2 here (and decrementing x5 by 1);
            // x3 is advanced in finalize().

            ; cmp x5, #0
            ; b.ne #32 // -> jump to loop body

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

            // Load V0/1/2.S4 with X/Y/Z values, post-increment
            //
            // We're actually loading two f32s, but we can pretend they're
            // doubles in order to move 64 bits at a time
            ; fmov s6, #1.0
            ; ldr s0, [x0], #4
            ; mov v0.S[1], v6.S[0]
            ; ldr s1, [x1], #4
            ; mov v1.S[2], v6.S[0]
            ; ldr s2, [x2], #4
            ; mov v2.S[3], v6.S[0]
            ; sub x5, x5, #1 // We handle 1 item at a time

            // Math begins below!
        );

        Self(out, loop_start)
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; ldr Q(reg(dst_reg)), [sp, #(sp_offset)]
        )
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
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
            ; ldr S(reg(out_reg)), [x3, #(src_arg * 4)]
        );
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

    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x4
            ; str Q(reg(out_reg)), [x4], #16
        );
        let jump_size: i32 = (self.0.ops.len() - self.1).try_into().unwrap();
        assert!(jump_size.abs() < (1 << 25));
        dynasm!(self.0.ops
           ; b #-jump_size
        );

        self.0.ops.finalize()
    }
}

#[cfg(target_arch = "x86_64")]
impl AssemblerT for GradSliceAssembler {
    type Data = Grad;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        unimplemented!()
    }
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        unimplemented!()
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        unimplemented!()
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        unimplemented!()
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        unimplemented!()
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        unimplemented!()
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        unimplemented!()
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        unimplemented!()
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        unimplemented!()
    }
}

impl SimdAssembler for GradSliceAssembler {
    const SIMD_SIZE: usize = 1;
}

////////////////////////////////////////////////////////////////////////////////

pub type JitGradSliceEval = JitBulkEval<GradSliceAssembler>;
