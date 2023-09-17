use super::{set_choice_bit, set_choice_exclusive};
use crate::{
    jit::{
        arch, point::PointAssembler, reg, AssemblerT, IMM_REG, OFFSET,
        REGISTER_LIMIT, SCRATCH_REG,
    },
    vm::ChoiceIndex,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation for the single-point assembler on `aarch64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register | Type                  |
/// |------------|----------|-----------------------|
/// | code       | `x0`     | `*const c_void`       |
/// | X          | `s0`     | `f32`                 |
/// | Y          | `s1`     | `f32`                 |
/// | Z          | `s2`     | `f32`                 |
/// | `vars`     | `x1`     | `*const f32` (array)  |
/// | `choices`  | `x2`     | `*const u8` (array)   |
/// | `simplify` | `x3`     | `*const u32` (single) |
impl<'a, D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>>
    AssemblerT<'a, D> for PointAssembler<'a, D>
{
    type T = f32;

    fn new(ops: &'a mut D) -> Self {
        Self(ops)
    }

    fn build_entry_point(
        ops: &'a mut D,
        slot_count: usize,
        _choice_array_size: usize,
    ) -> usize {
        let offset = ops.offset().0;
        let mem_offset = Self::function_entry(ops, slot_count);
        let out_reg = 0;
        dynasm!(ops
            // Jump into threaded code
            ; ldr x15, [x0, #0]
            ; blr x15
            // Return from threaded code here

            // Prepare our return value
            ; fmov  s0, S(reg(out_reg))
        );
        arch::function_exit(ops, mem_offset);
        offset
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT || reg(dst_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(src_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0 ; ldr S(reg(dst_reg)), [sp, #(sp_offset)])
    }

    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT || reg(src_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(dst_mem);
        assert!(sp_offset <= 16384);
        dynasm!(self.0 ; str S(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0 ; fmov S(reg(out_reg)), S(src_arg as u32));
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0
            ; ldr S(reg(out_reg)), [x1, #(src_arg * 4)]
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fmov S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fneg S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fabs S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            ; fmov s7, #1.0
            ; fdiv S(reg(out_reg)), s7, S(reg(lhs_reg))
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fsqrt S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(lhs_reg)))
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fadd S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fsub S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fdiv S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi >R
            ; b.gt >L

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmax S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b >E

            ; R:
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            ; b >E

            ; L:
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            // fall-through to end

            // <- end
            ; E:
        );
    }

    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.mi >L
            ; b.gt >R

            // Equal or NaN; do the comparison to collapse NaNs
            ; fmin S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b >E // -> end

            ; L: // LHS
            ; fmov S(reg(out_reg)), S(reg(lhs_reg))
            ; b >E

            ; R: // RHS
            ; fmov S(reg(out_reg)), S(reg(rhs_reg))
            // fall-through to end

            ; E:
        );
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; fmov S(IMM_REG as u32), w9
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    /// Uses `v4`, `v5`, `x14`, `x15`
    fn build_min_reg_reg_choice(
        &mut self,
        inout_reg: u8,
        arg_reg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        dynasm!(self.0
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x2, #i]
            // Jump to V if the choice bit was previously set
            ; tst w15, #1
            ; b.ne >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, then branch to the end
            ; fmov S(reg(inout_reg)), S(reg(arg_reg))
            ; b >E

            ; V: // There was a previous value, so we have to do the comparison
            ; fcmp S(reg(inout_reg)), S(reg(arg_reg))
            ; b.mi >L
            ; b.gt >R

            //////////////////////////////////////////////////////////////////
            // Fallthrough: ambiguous case
            // Equal or NaN; do the comparison to collapse NaNs
            ; fmin S(reg(inout_reg)), S(reg(inout_reg)), S(reg(arg_reg))
        );
        set_choice_bit(self.0, choice);
        dynasm!(self.0
            ; b >E // -> end
            // end of ambiguous case
            //////////////////////////////////////////////////////////////////

            ; L: // inout is smaller, so write the simplify bit
            ; mov w15, 0x1
            ; str w15, [x3]
            ; b >E

            ; R: // arg is smaller, so write simplify + choice bit
            ; mov w15, 0x1
            ; str w15, [x3]
            ; fmov S(reg(inout_reg)), S(reg(arg_reg)) // copy the reg
        );
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            // end of arg-smaller case (R label); fallthrough to end
            //////////////////////////////////////////////////////////////////

            ; E: // end branch label
            // Set choice bit 0 and write it back to memory

            // TODO: this adds an extra load/store, but tracking it would be
            // annoying.
            ; ldrb w15, [x2, #i]
            ; orr w15, w15, #1
            ; strb w15, [x2, #i]
        );
    }

    fn build_min_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        // V6 doesn't conflict with registers used in `build_min_reg_reg_choice`
        let lhs = SCRATCH_REG.wrapping_sub(OFFSET);
        self.build_load(lhs, mem);
        self.build_min_reg_reg_choice(lhs, arg, choice);
        self.build_store(mem, lhs);
    }

    fn build_min_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_min_mem_reg_choice(mem, rhs, choice);
    }

    fn build_min_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_min_reg_reg_choice(reg, rhs, choice);
    }

    fn build_max_reg_reg_choice(
        &mut self,
        inout_reg: u8,
        arg_reg: u8,
        choice: ChoiceIndex,
    ) {
        // basically the same as min_reg_reg_choice
        let i = choice.index as u32;
        dynasm!(self.0
            //  Bit 0 of the choice indicates whether it has a value
            ; ldrb w15, [x2, #i]
            // Jump to V if the choice bit was previously set
            ; tst w15, #1 // Sets the Z(ero) flag if the bit is not set
            ; b.ne >V     // `ne` is true when the Z flag is not set

            // Fallthrough: there was no value, so we set it here
            // Copy the value, then branch to the end
            ; fmov S(reg(inout_reg)), S(reg(arg_reg))
            ; b >E

            ; V: // There was a previous value, so we have to do the comparison
            ; fcmp S(reg(inout_reg)), S(reg(arg_reg))
            ; b.mi >R
            ; b.gt >L

            //////////////////////////////////////////////////////////////////
            // Fallthrough: ambiguous case
            // Equal or NaN; do the comparison to collapse NaNs
            ; fmax S(reg(inout_reg)), S(reg(inout_reg)), S(reg(arg_reg))
        );
        set_choice_bit(self.0, choice);
        dynasm!(self.0
            ; b >E // -> end
            // end of ambiguous case
            //////////////////////////////////////////////////////////////////

            ; L: // inout is larger, so write the simplify bit
            ; mov w15, #1
            ; str w15, [x3]
            ; b >E

            ; R: // arg is larger, so write simplify + choice bit
            ; mov w15, #1
            ; str w15, [x3]
            ; fmov S(reg(inout_reg)), S(reg(arg_reg)) // copy the reg
        );
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            // end of arg-larger case (R label); fallthrough to end
            //////////////////////////////////////////////////////////////////

            ; E: // end branch label
            // Set choice bit 0 and write it back to memory
            // TODO: this adds an extra load/store, but tracking it would be
            // annoying.
            ; ldrb w15, [x2, #i]
            ; orr w15, w15, #1
            ; strb w15, [x2, #i]
        );
    }

    fn build_max_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        // V6 doesn't conflict with registers used in `build_max_reg_reg_choice`
        let lhs = 6u8.wrapping_sub(OFFSET);
        self.build_load(lhs, mem);
        self.build_max_reg_reg_choice(lhs, arg, choice);
        self.build_store(mem, lhs);
    }

    fn build_max_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_max_mem_reg_choice(mem, rhs, choice);
    }

    fn build_max_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_max_reg_reg_choice(reg, rhs, choice);
    }

    fn build_copy_reg_reg_choice(
        &mut self,
        out: u8,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(self.0
            ; fmov S(reg(out)), S(reg(arg))
            ; mov w15, #3
            ; strb w15, [x2, #i]
        );
    }

    fn build_copy_imm_reg_choice(
        &mut self,
        out: u8,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_copy_reg_reg_choice(out, rhs, choice);
    }

    fn build_copy_imm_mem_choice(
        &mut self,
        out: u32,
        imm: f32,
        choice: ChoiceIndex,
    ) {
        let rhs = self.load_imm(imm);
        self.build_copy_reg_mem_choice(out, rhs, choice);
    }

    fn build_copy_reg_mem_choice(
        &mut self,
        out: u32,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(self.0
            ; mov w15, #3 // bit 1 and bit 0, to mark the choice as present
            ; strb w15, [x2, #i]
        );
        self.build_store(out, arg);
    }
}
