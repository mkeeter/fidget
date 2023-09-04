use crate::jit::{
    float_slice::FloatSliceAssembler, mmap::Mmap, reg, AssemblerData,
    AssemblerT, Error, IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub const SIMD_WIDTH: usize = 4;

/// Assembler for SIMD point-wise evaluation on `aarch64`
///
/// | Argument  | Register | Type                       |
/// | ----------|----------|----------------------------|
/// | code      | `x0`     | `*const void*`             |
/// | X         | `x1`     | `*const [f32; 4]` (array)  |
/// | Y         | `x2`     | `*const [f32; 4]` (array)  |
/// | Z         | `x3`     | `*const [f32; 4]` (array)  |
/// | vars      | `x4`     | `*const f32` (array)       |
/// | out       | `x5`     | `*mut [f32; 4]` (array)    |
/// | size      | `x6`     | `u64`                      |
/// | `choices` | `x7`     | `*const u8` (array)        |
///
/// The arrays (other than `vars`) must be an even multiple of 4 floats, since
/// we're using NEON and 128-bit wide operations for everything.  The `vars`
/// array contains single `f32` values, which are broadcast into SIMD registers
/// when they are used.
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`
#[cfg(target_arch = "aarch64")]
impl AssemblerT for FloatSliceAssembler {
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
            ; mov x8, x0 // back up X0 in X8 (since we're not using XR)

        );
        out.0.prepare_stack(slot_count);

        dynasm!(out.0.ops
            // The loop returns here, and we check whether we need to loop
            ; ->L:
            // Remember, at this point we have
            //  x1: x input array pointer (advancing)
            //  x2: y input array pointer (advancing)
            //  x3: z input array pointer (advancing)
            //  x4: vars input array pointer (non-advancing)
            //  x5: output array pointer (advancing)
            //  x6: number of points to evaluate
            //  x7: array of choice data
            //  x8: backup value for x0's starting point
            //
            // We'll be advancing x1, x2, x3 here (and decrementing x6 by 4);

            ; cmp x5, #0
            ; b.ne >P // skip to loop body

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

            // Clear out the choices array, operating on 64-bit chunks
            // (since that's our guaranteed minimum alignment)
            //
            // TODO: this would be more efficient if we did one byte (or bit)
            // per choice, instead of clearing the entire array, since we only
            // need to track "is this choice active"
            ; mov x9, #choice_array_size as u64
            ; mov x10, #0
            ; mov x11, x7
            ; cmp x9, #0
            ; ->memclr:
            ; b.eq >O
            ; sub x9, x9, 1
            ; str x10, [x11], #8
            ; b ->memclr

            // Call into threaded code
            ; O:
            ; mov x0, x8
            ; ldr x15, [x0, #0]
            ; blr x15
            // Return from threaded code

            // Prepare our return value, writing to the pointer in x5
            ; str Q(reg(out_reg)), [x5], #16
            ; b ->L
        );
        out.finalize().unwrap()
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
            ; ldr w15, [x4, #(src_arg * 4)]
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
        // V6 doesn't conflict with registers used in `build_min_reg_reg_choice`
        let lhs = 6u8.wrapping_sub(OFFSET);
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
        // V6 doesn't conflict with registers used in `build_max_reg_reg_choice`
        let lhs = 6u8.wrapping_sub(OFFSET);
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
        let i = choice.index as u32;
        dynasm!(self.0.ops
            //  Bit 0 of the choice indicates whether it has a value
            ; ldr b15, [x7, #i]
            // Jump to V if the choice bit was previously set
            ; ands w15, w15, #1
            ; b.eq >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; mov w15, #1
            ; str b15, [x7]
            ; b >E

            ; V:
            ; fmin V(reg(inout_reg)).s4, V(reg(inout_reg)).s4, V(reg(arg_reg)).s4

            ; E:
        );
    }

    fn build_max_reg_reg_choice(
        &mut self,
        inout_reg: u8,
        arg_reg: u8,
        choice: crate::vm::ChoiceIndex,
    ) {
        let i = choice.index as u32;
        dynasm!(self.0.ops
            //  Bit 0 of the choice indicates whether it has a value
            ; ldr b15, [x7, #i]
            // Jump to V if the choice bit was previously set
            ; ands w15, w15, #1
            ; b.eq >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; mov w15, #1
            ; str b15, [x7]
            ; b >E

            ; V:
            ; fmax V(reg(inout_reg)).s4, V(reg(inout_reg)).s4, V(reg(arg_reg)).s4

            ; E:
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
            ; str b15, [x7, #i]
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
            ; str b15, [x7, #i]
        );
        self.build_store(out, arg);
    }

    fn build_jump(&mut self) {
        crate::jit::arch::build_jump(&mut self.0.ops)
    }
}
