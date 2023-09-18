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
impl<D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>> AssemblerT<D>
    for PointAssembler
{
    type T = f32;

    fn build_entry_point(
        ops: &mut D,
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
    fn build_load(ops: &mut D, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT || reg(dst_reg) == SCRATCH_REG as u32);
        let sp_offset = <Self as AssemblerT<D>>::stack_pos(src_mem);
        assert!(sp_offset <= 16384);
        dynasm!(ops ; ldr S(reg(dst_reg)), [sp, #(sp_offset)])
    }

    /// Writes from `src_reg` to `dst_mem`
    fn build_store(ops: &mut D, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT || reg(src_reg) == SCRATCH_REG as u32);
        let sp_offset = <Self as AssemblerT<D>>::stack_pos(dst_mem);
        assert!(sp_offset <= 16384);
        dynasm!(ops ; str S(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(ops: &mut D, out_reg: u8, src_arg: u8) {
        dynasm!(ops ; fmov S(reg(out_reg)), S(src_arg as u32));
    }
    fn build_var(ops: &mut D, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(ops
            ; ldr S(reg(out_reg)), [x1, #(src_arg * 4)]
        );
    }
    fn build_copy(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops ; fmov S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_neg(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops ; fneg S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_abs(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops ; fabs S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_recip(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops
            ; fmov s7, #1.0
            ; fdiv S(reg(out_reg)), s7, S(reg(lhs_reg))
        )
    }
    fn build_sqrt(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops ; fsqrt S(reg(out_reg)), S(reg(lhs_reg)))
    }
    fn build_square(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        dynasm!(ops ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(lhs_reg)))
    }
    fn build_add(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
            ; fadd S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_sub(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
            ; fsub S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_mul(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
            ; fmul S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_div(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
            ; fdiv S(reg(out_reg)), S(reg(lhs_reg)), S(reg(rhs_reg))
        )
    }
    fn build_max(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
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

    fn build_min(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(ops
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
    fn load_imm(ops: &mut D, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(ops
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; fmov S(IMM_REG as u32), w9
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    /// Uses `v4`, `v5`, `x14`, `x15`
    fn build_min_reg_reg_choice(
        ops: &mut D,
        inout_reg: u8,
        arg_reg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        dynasm!(ops
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
        set_choice_bit(ops, choice);
        dynasm!(ops
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
        set_choice_exclusive(ops, choice);
        dynasm!(ops
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

    fn build_max_reg_reg_choice(
        ops: &mut D,
        inout_reg: u8,
        arg_reg: u8,
        choice: ChoiceIndex,
    ) {
        // basically the same as min_reg_reg_choice
        let i = choice.index as u32;
        dynasm!(ops
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
        set_choice_bit(ops, choice);
        dynasm!(ops
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
        set_choice_exclusive(ops, choice);
        dynasm!(ops
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

    fn build_copy_reg_reg_choice(
        ops: &mut D,
        out: u8,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(ops
            ; fmov S(reg(out)), S(reg(arg))
            ; mov w15, #3
            ; strb w15, [x2, #i]
        );
    }

    fn build_copy_reg_mem_choice(
        ops: &mut D,
        out: u32,
        arg: u8,
        choice: ChoiceIndex,
    ) {
        let i = choice.index as u32;
        assert_eq!(choice.bit, 1);
        dynasm!(ops
            ; mov w15, #3 // bit 1 and bit 0, to mark the choice as present
            ; strb w15, [x2, #i]
        );
        Self::build_store(ops, out, arg);
    }
}
