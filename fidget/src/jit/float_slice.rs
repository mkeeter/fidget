use crate::jit::{
    mmap::Mmap, reg, AssemblerData, AssemblerT, JitBulkEval, SimdAssembler,
    IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi};

/// Assembler for SIMD point-wise evaluation.
///
/// Arguments are passed as 3x `*const f32` in `x0-2`, a var array in
/// `x3`, and an output array `*mut f32` in `x4`.  Each pointer in the input
/// and output arrays represents 4x `f32`; the var array is single `f32`s
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`
pub struct FloatSliceAssembler(AssemblerData<[f32; 4]>, usize);

#[cfg(target_arch = "aarch64")]
impl AssemblerT for FloatSliceAssembler {
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
            ; ldp d0, d1, [x0], #16
            ; mov v0.d[1], v1.d[0]
            ; ldp d1, d2, [x1], #16
            ; mov v1.d[1], v2.d[0]
            ; ldp d2, d3, [x2], #16
            ; mov v2.d[1], v3.d[0]
            ; sub x5, x5, #4 // We handle 4 items at a time
        );

        Self(out, loop_start)
    }
    /// Reads from `src_mem` to `dst_reg`, using D4 as an intermediary
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem);
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
        let sp_offset = self.0.stack_pos(dst_mem);
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
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0.ops
            ; ldr w15, [x3, #(src_arg * 4)]
            ; dup V(reg(out_reg)).s4, w15
        );
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
    fn build_fma(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmla V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
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

    fn finalize(mut self, out_reg: u8) -> Mmap {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x4
            // It's fine to overwrite X at this point in V0, since we're not
            // using it anymore.
            ; mov v0.d[0], V(reg(out_reg)).d[1]
            ; stp D(reg(out_reg)), d0, [x4], #16
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
impl AssemblerT for FloatSliceAssembler {
    type Data = f32;

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
    fn build_fma(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
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
    fn finalize(mut self, out_reg: u8) -> Mmap {
        unimplemented!()
    }
}

impl SimdAssembler for FloatSliceAssembler {
    const SIMD_SIZE: usize = 4;
}

////////////////////////////////////////////////////////////////////////////////

pub type JitFloatSliceEval = JitBulkEval<FloatSliceAssembler>;
