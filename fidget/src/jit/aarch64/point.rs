use crate::{
    jit::{
        mmap::Mmap, point::PointAssembler, reg, AssemblerData, AssemblerT,
        CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT, IMM_REG, OFFSET,
        REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi};

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
