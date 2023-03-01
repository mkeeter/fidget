use crate::{
    jit::{
        mmap::Mmap, reg, AssemblerData, AssemblerT, JitTracingEval,
        CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT, IMM_REG, OFFSET,
        REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi};

#[cfg(target_arch = "x86_64")]
use dynasmrt::DynasmLabelApi;

pub struct PointAssembler(AssemblerData<f32>);

#[cfg(target_arch = "aarch64")]
impl AssemblerT for PointAssembler {
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

        Self(out)
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; ldr S(reg(dst_reg)), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0.ops ; str S(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; fmov S(reg(out_reg)), S(src_arg as u32));
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0.ops
            ; ldr S(reg(out_reg)), [x0, #(src_arg * 4)]
        );
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
            ; ldrb w14, [x1]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi #20 // -> RHS
            ; b.gt #32 // -> LHS

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmax S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_BOTH
            ; b #32 // -> end

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; strb w14, [x2, #0] // write a non-zero value to simplify
            ; b #16

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; strb w14, [x2, #0] // write a non-zero value to simplify
            // fall-through to end

            // <- end
            ; strb w14, [x1], #1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; ldrb w14, [x1]
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi #20
            ; b.gt #32

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmin S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_BOTH
            ; b #32 // -> end

            // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; strb w14, [x2, #0] // write a non-zero value to simplify
            ; b #16

            // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; strb w14, [x2, #0]
            // fall-through to end

            // <- end
            ; strb w14, [x1], #1 // post-increment
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

        self.0.ops.finalize()
    }
}

/// Registers are passed in as follows
/// - X, Y, Z are in `xmm0-2`
/// - `vars` is in `rdi`
/// - `choices` is in `rsi`
/// - `simplify` is in `rdx`
#[cfg(target_arch = "x86_64")]
impl AssemblerT for PointAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
            ; vzeroupper
            // Put X/Y/Z on the stack so we can use those registers
            ; vmovss [rbp - 4], xmm0
            ; vmovss [rbp - 8], xmm1
            ; vmovss [rbp - 12], xmm2
        );
        out.prepare_stack(slot_count);
        Self(out)
    }

    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(src_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovss Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(dst_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovss [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops
            // Pull X/Y/Z from the stack, where they've been placed by init()
            ; vmovss Rx(reg(out_reg)), [rbp - 4 * (src_arg as i32 + 1)]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), [rdi + 4 * (src_arg as i32)]
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        // Flip the sign bit in the float
        dynasm!(self.0.ops
            // TODO: build this in xmm0 directly
            ; mov eax, 0x80000000u32 as i32
            ; movd xmm0, eax
            ; vxorps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        // Clear the sign bit in the float
        dynasm!(self.0.ops
            ; mov eax, 0x7fffffffu32 as i32
            ; vmovd xmm0, eax
            ; vandps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        )
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        let imm = self.load_imm(1.0);
        dynasm!(self.0.ops
            ; divss Rx(reg(imm)), Rx(reg(lhs_reg))
            ; movss Rx(reg(out_reg)), Rx(reg(imm))
        );
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; sqrtss Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vsubss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vdivss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N
            ; ja >L
            ; jb >R

            // Fallthrough for equal, so just copy to the output register
            ; or [rsi], CHOICE_BOTH as i8
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            // Fallthrough for NaN, which are !=; do a float addition to
            // propagate it to the output register.
            ; N:
            ; or [rsi], CHOICE_BOTH as i8
            // TODO: this can't be the best way to make a NAN
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jmp >O

            ; L:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or [rsi], CHOICE_LEFT as i8
            ; or [rdx], 1
            ; jmp >O

            ; R:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or [rsi], CHOICE_RIGHT as i8
            ; or [rdx], 1
            // fallthrough to out

            ; O:
        );
        self.0.ops.commit_local().unwrap()
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N
            ; ja >R
            ; jb >L

            // Fallthrough for equal, so just copy to the output register
            ; or [rsi], CHOICE_BOTH as i8
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            ; N:
            ; or [rsi], CHOICE_BOTH as i8
            // TODO: this can't be the best way to make a NAN
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jmp >O

            ; L:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or [rsi], CHOICE_LEFT as i8
            ; or [rdx], 1
            ; jmp >O

            ; R:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or [rsi], CHOICE_RIGHT as i8
            ; or [rdx], 1
            // fallthrough to out

            ; O:
        );
        self.0.ops.commit_local().unwrap()
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; mov eax, imm_u32 as i32
            ; vmovd Rx(IMM_REG), eax
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Prepare our return value
            ; vmovss xmm0, xmm0, Rx(reg(out_reg))
            ; add rsp, self.0.mem_offset as i32
            ; pop rbp
            ; emms
            ; ret
        );
        self.0.ops.finalize()
    }
}

pub type JitPointEval = JitTracingEval<PointAssembler>;
