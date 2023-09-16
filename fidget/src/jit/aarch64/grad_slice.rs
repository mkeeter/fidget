use crate::{
    jit::{
        grad_slice::GradSliceAssembler, mmap::Mmap, reg, AssemblerData,
        AssemblerT, IMM_REG, OFFSET, REGISTER_LIMIT, SCRATCH_REG,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation for the gradient slice assembler on `aarch64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register | Type                |
/// |------------|----------|---------------------|
/// | code       | `x0`     | `*const void*`      |
/// | X          | `x1`     | `*const f32`        |
/// | Y          | `x2`     | `*const f32`        |
/// | Z          | `x3`     | `*const f32`        |
/// | `vars`     | `x4`     | `*const f32`        |
/// | `out`      | `x5`     | `*const [f32; 4]`   |
/// | `count`    | `x6`     | `u64`               |
/// | `choices`  | `x7`     | `*const u8` (array) |
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`.  Each SIMD register
/// is in the order `[value, dx, dy, dz]`, e.g. the value for X is in `V0.S0`.
impl AssemblerT for GradSliceAssembler {
    fn new() -> Self {
        Self(AssemblerData::new())
    }

    fn build_entry_point(slot_count: usize, choice_array_size: usize) -> Mmap {
        let mut out = Self::new();
        let out_reg = 0;
        dynasm!(out.0.ops
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
        out.0.prepare_stack(slot_count);

        dynasm!(out.0.ops
            // The loop returns here, and we check whether we need to loop
            ; ->L:
            // Remember, at this point we have
            //  x1: x input array pointer
            //  x2: y input array pointer
            //  x3: z input array pointer
            //  x4: vars input array pointer (non-advancing)
            //  x5: output array pointer
            //  x6: number of points to evaluate
            //
            // We'll be advancing x1, x2, x3 here (and decrementing x6 by 1);
            // x6 is advanced in finalize().

            ; cmp x6, #0
            ; b.ne >P // -> jump to loop body

            // fini:
            // This is our finalization code, which happens after all evaluation
            // is complete.
            //
            // Restore stack space used for spills
            ; add   sp, sp, #(out.0.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret

            ; P:
            // Load V0/1/2.S4 with X/Y/Z values, post-increment
            //
            // We're actually loading two f32s, but we can pretend they're
            // doubles in order to move 64 bits at a time
            ; fmov s6, #1.0
            ; ldr s0, [x1], #4
            ; mov v0.S[1], v6.S[0]
            ; ldr s1, [x2], #4
            ; mov v1.S[2], v6.S[0]
            ; ldr s2, [x3], #4
            ; mov v2.S[3], v6.S[0]
            ; sub x6, x6, #1 // We handle 1 item at a time

            // Clear out the choices array, operating on 64-bit chunks
            // (since that's our guaranteed minimum alignment)
            //
            // TODO: this would be more efficient if we did one byte (or bit)
            // per choice, instead of clearing the entire array, since we only
            // need to track "is this choice active"
            ; mov x9, #choice_array_size as u64
            ; mov x10, x7

            ; ->memclr:
            ; cmp x9, #0
            ; b.eq >O
            ; sub x9, x9, 1
            ; str xzr, [x10], #8 // post-increment
            ; b ->memclr

            // Call into threaded code
            ; O:
            ; mov x8, x0 // save x0
            ; ldr x15, [x0, #0]
            ; blr x15
            ; mov x0, x8 // restore x0
            // Return from threaded code

            // Prepare our return value, writing to the pointer in x5
            ; str Q(reg(out_reg)), [x5], #16 // post-increment
            ; b ->L // Jump back to the loop start
        );
        out.finalize().unwrap()
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT || reg(dst_reg) == SCRATCH_REG as u32);
        let sp_offset = self.0.stack_pos(src_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0.ops
            ; ldr Q(reg(dst_reg)), [sp, #(sp_offset)]
        )
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT || reg(src_reg) == SCRATCH_REG as u32);
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
            ; ldr S(reg(out_reg)), [x4, #(src_arg * 4)]
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
            ; b.gt >Lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b >End // -> end
            ; Lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; End:
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.lt >Lhs // -> lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b >End // -> end
            ; Lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            ; End:
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

    fn finalize(self) -> Result<Mmap, Error> {
        self.0.ops.try_into()
    }

    fn build_min_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_min_mem_reg_choice(mem, rhs, choice);
    }

    fn build_max_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_max_mem_reg_choice(mem, rhs, choice);
    }

    fn build_min_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_min_reg_reg_choice(reg, rhs, choice);
    }

    fn build_max_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_max_reg_reg_choice(reg, rhs, choice);
    }

    fn build_min_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        let lhs = SCRATCH_REG.wrapping_sub(OFFSET);
        self.build_load(lhs, mem);
        self.build_min_reg_reg_choice(lhs, arg, choice);
        self.build_store(mem, lhs);
    }

    fn build_max_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        let lhs = SCRATCH_REG.wrapping_sub(OFFSET);
        self.build_load(lhs, mem);
        self.build_max_reg_reg_choice(lhs, arg, choice);
        self.build_store(mem, lhs);
    }

    fn build_min_reg_reg_choice(
        &mut self,
        inout_reg: u8,
        arg_reg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        // Note: we can't use SCRATCH_REG (v6) here, because it may be our inout
        let i = choice.index as u32;
        dynasm!(self.0.ops
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x7, #i]
            // Jump to V if the choice bit was previously set
            ; tst w15, #1
            ; b.ne >Compare

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; b >End

            ; Compare:
            ; fcmp S(reg(inout_reg)), S(reg(arg_reg))
            ; b.lt >End
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16

            ; End:
            ; mov w15, #1
            ; strb w15, [x7, #i]
        );
    }

    fn build_max_reg_reg_choice(
        &mut self,
        inout_reg: u8,
        arg_reg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        // Note: we can't use SCRATCH_REG (v6) here, because it may be our inout
        let i = choice.index as u32;
        dynasm!(self.0.ops
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x7, #i]
            // Jump to Compare if the choice bit was previously set
            ; tst w15, #1
            ; b.ne >Compare

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; b >End

            ; Compare:
            ; fcmp S(reg(inout_reg)), S(reg(arg_reg))
            ; b.gt >End
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16

            ; End:
            ; mov w15, #1
            ; strb w15, [x7, #i]
        );
    }

    fn build_copy_imm_reg_choice(
        &mut self,
        out: u8,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_copy_reg_reg_choice(out, rhs, choice);
    }

    fn build_copy_imm_mem_choice(
        &mut self,
        out: u32,
        imm: f32,
        choice: crate::vm::ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_copy_reg_mem_choice(out, rhs, choice);
    }

    fn build_copy_reg_reg_choice(
        &mut self,
        out: u8,
        arg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(self.0.ops
            ; mov V(reg(out)).b16, V(reg(arg)).b16
            ; mov w15, #3
            ; strb w15, [x7, #i]
        );
    }

    fn build_copy_reg_mem_choice(
        &mut self,
        out: u32,
        arg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(self.0.ops
            ; mov w15, #3
            ; strb w15, [x7, #i]
        );
        self.build_store(out, arg);
    }

    fn build_jump(&mut self) {
        crate::jit::arch::build_jump(&mut self.0.ops)
    }
}
