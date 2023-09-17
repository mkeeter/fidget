use crate::jit::{
    arch, float_slice::FloatSliceAssembler, reg, AssemblerT, IMM_REG, OFFSET,
    REGISTER_LIMIT, SCRATCH_REG,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub const SIMD_WIDTH: usize = 4;

/// Assembler for SIMD point-wise evaluation on `aarch64`
///
/// | Argument | Register | Type                       |
/// | ---------|----------|----------------------------|
/// | code     | `x0`     | `*const void*`             |
/// | X        | `x1`     | `*const [f32; 4]` (array)  |
/// | Y        | `x2`     | `*const [f32; 4]` (array)  |
/// | Z        | `x3`     | `*const [f32; 4]` (array)  |
/// | vars     | `x4`     | `*const f32` (array)       |
/// | out      | `x5`     | `*mut [f32; 4]` (array)    |
/// | size     | `x6`     | `u64`                      |
/// | choices  | `x7`     | `*mut u8` (array)          |
///
/// The X/Y/Z arrays must be an even multiple of 4 floats, since we're using
/// NEON and 128-bit wide operations for everything.  The `vars` array contains
/// single `f32` values, which are broadcast into SIMD registers when they are
/// used.
///
/// During evaluation, X, Y, and Z are stored in `V{0,1,2}.S4`
#[cfg(target_arch = "aarch64")]
impl<'a, D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>>
    AssemblerT<'a, D> for FloatSliceAssembler<'a, D>
{
    type T = [f32; 4];

    fn new(ops: &'a mut D) -> Self {
        Self(ops)
    }

    fn build_entry_point(
        ops: &'a mut D,
        slot_count: usize,
        choice_array_size: usize,
    ) -> usize {
        let offset = ops.offset().0;
        let mem_offset = Self::function_entry(ops, slot_count);
        let out_reg = 0;

        dynasm!(ops
            // The loop returns here, and we check whether we need to loop
            ; ->float_loop:
            // Remember, at this point we have
            //  x1: x input array pointer (advancing)
            //  x2: y input array pointer (advancing)
            //  x3: z input array pointer (advancing)
            //  x4: vars input array pointer (non-advancing)
            //  x5: output array pointer (advancing)
            //  x6: number of points to evaluate (decreasing)
            //  x7: array of choice data (non-advancing, memclr'd)
            //
            // We'll be advancing x1, x2, x3 here (and decrementing x6 by 4);

            ; cmp x6, #0
            ; b.eq >Exit // we're done!

            // Fallthrough into loop body:
            //
            // Load V0/1/2.S4 with X/Y/Z values, post-increment
            ; ldr q0, [x1], #16
            ; ldr q1, [x2], #16
            ; ldr q2, [x3], #16
            ; sub x6, x6, #4 // We handle 4 items at a time

            // Clear out the choices array, operating on 64-bit chunks
            // (since that's our guaranteed minimum alignment)
            //
            // TODO: this would be more efficient if we did one byte (or bit)
            // per choice, instead of clearing the entire array, since we only
            // need to track "is this choice active"
            ; mov x9, #choice_array_size as u64
            ; mov x10, x7

            ; ->float_memclr:
            ; cmp x9, #0
            ; b.eq >O
            ; sub x9, x9, 1
            ; str xzr, [x10], 8 // post-increment
            ; b ->float_memclr

            // Call into threaded code
            ; O:
            ; mov x8, x0 // store x0 (start of threaded code)
            ; ldr x15, [x0, #0]
            ; blr x15
            // Return from threaded code
            ; mov x0, x8 // restore x0

            // Prepare our return value, writing to the pointer in x5
            ; str Q(reg(out_reg)), [x5], #16
            ; b ->float_loop

            ; Exit:
        );
        arch::function_exit(ops, mem_offset);
        offset
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT || reg(dst_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(src_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0
            ; ldr Q(reg(dst_reg)), [sp, #(sp_offset)]
        )
    }

    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT || reg(src_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(dst_mem);
        assert!(sp_offset < 65536);
        dynasm!(self.0
            ; str Q(reg(src_reg)), [sp, #(sp_offset)]
        )
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0 ; mov V(reg(out_reg)).b16, V(src_arg as u32).b16);
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0
            ; ldr w15, [x4, #(src_arg * 4)]
            ; dup V(reg(out_reg)).s4, w15
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16)
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fneg V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fabs V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            ; fmov s7, #1.0
            ; dup v7.s4, v7.s[0]
            ; fdiv V(reg(out_reg)).s4, v7.s4, V(reg(lhs_reg)).s4
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fsqrt V(reg(out_reg)).s4, V(reg(lhs_reg)).s4)
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            ; fmul V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(lhs_reg)).s4
        )
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fadd V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fsub V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fmul V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fdiv V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fmax V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fmin V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
        )
    }

    /// Loads an immediate into register V4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; dup V(IMM_REG as u32).s4, w9
        );
        IMM_REG.wrapping_sub(OFFSET)
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
        // V6 doesn't conflict with registers used in `build_max_reg_reg_choice`
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
        assert!(i < 4096);
        dynasm!(self.0
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x7, #i]
            // Jump to V if the choice bit was previously set
            ; tst w15, #1
            ; b.ne >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; b >E

            ; V:
            ; fmin V(reg(inout_reg)).s4, V(reg(inout_reg)).s4, V(reg(arg_reg)).s4

            ; E:
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
        assert!(i < 4096);
        dynasm!(self.0
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x7, #i]
            // Jump to V if the choice bit was previously set
            ; tst w15, #1
            ; b.ne >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, write the choice bit, then jump to the end
            ; mov V(reg(inout_reg)).b16, V(reg(arg_reg)).b16
            ; b >E

            ; V:
            ; fmax V(reg(inout_reg)).s4, V(reg(inout_reg)).s4, V(reg(arg_reg)).s4

            ; E:
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
        dynasm!(self.0
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
        dynasm!(self.0
            ; mov w15, #3
            ; strb w15, [x7, #i]
        );
        self.build_store(out, arg);
    }
}
