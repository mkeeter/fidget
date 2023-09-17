use crate::{
    eval::types::Interval,
    jit::{
        arch::{self, set_choice_bit, set_choice_exclusive},
        interval::IntervalAssembler,
        reg, AssemblerT, IMM_REG, OFFSET, REGISTER_LIMIT, SCRATCH_REG,
    },
    vm::ChoiceIndex,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation for the interval assembler on `aarch64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register   | Type                    |
/// |------------|------------|-------------------------|
/// | code       | `x0`       | `*const c_void`         |
/// | X          | `(s0, s1)` | `(f32, f32)`            |
/// | Y          | `(s2, s3)` | `(f32, f32)`            |
/// | Z          | `(s4, s5)` | `(f32, f32)`            |
/// | `vars`     | `x1`       | `*const f32` (array)    |
/// | `choices`  | `x2`       | `*const u8` (array)     |
/// | `simplify` | `x3`       | `*const u32` (single)   |
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S2`.  Each SIMD register
/// stores an interval.  `s[0]` is the lower bound of the interval and `s[1]` is
/// the upper bound; for example, `V0.S0` represents the lower bound for X.
impl<'a, D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>>
    AssemblerT<'a, D> for IntervalAssembler<'a, D>
{
    type T = Interval;

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
            // Arguments are passed in S0-5; collect them into V0-1
            ; mov v0.s[1], v1.s[0]
            ; mov v1.s[0], v2.s[0]
            ; mov v1.s[1], v3.s[0]
            ; mov v2.s[0], v4.s[0]
            ; mov v2.s[1], v5.s[0]

            // Jump into threaded code
            // TODO: this means that threaded code can't use the link register;
            // is that an issue?
            ; ldr x15, [x0, #0]
            ; blr x15
            // Return from threaded code here

            // Prepare our return value
            ; mov s0, V(reg(out_reg)).s[0]
            ; mov s1, V(reg(out_reg)).s[1]
        );
        arch::function_exit(ops, mem_offset);
        offset
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT || reg(dst_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(src_mem);
        assert!(sp_offset <= 32768);
        dynasm!(self.0 ; ldr D(reg(dst_reg)), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT || reg(src_reg) == SCRATCH_REG as u32);
        let sp_offset = Self::stack_pos(dst_mem);
        assert!(sp_offset <= 32768);
        dynasm!(self.0 ; str D(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0 ; fmov D(reg(out_reg)), D(src_arg as u32));
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0
            ; ldr w15, [x1, #(src_arg * 4)]
            ; dup V(reg(out_reg)).s2, w15
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0 ; fmov D(reg(out_reg)), D(reg(lhs_reg)))
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            ; fneg V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2
        )
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            // Store lhs < 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4

            // Store abs(lhs) in V(reg(out_reg))
            ; fabs V(reg(out_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne >L // -> upper_lz

            // Check whether lhs.lower < 0
            ; tst x15, #0x1

            // otherwise, we're good; return the original
            ; b.eq >E // -> end

            // if lhs.lower < 0, then the output is
            //  [0.0, max(abs(lower, upper))]
            ; movi d4, #0
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; fmov D(reg(out_reg)), d4
            // Fall through to do the swap

            ; L: // <- upper_lz
            // if upper < 0
            //   return [-upper, -lower]
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            ; E:// <- end
        );
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0
            // Check whether lhs.lower > 0.0
            ; fcmp S(reg(lhs_reg)), 0.0
            ; b.gt >O // -> okay

            // Check whether lhs.upper < 0.0
            ; mov s4, V(reg(lhs_reg)).s[1]
            ; fcmp s4, 0.0
            ; b.mi >O // -> okay

            // Bad case: the division spans 0, so return NaN
            ; movz w15, #(nan_u32 >> 16), lsl 16
            ; movk w15, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w15
            ; b >E // -> end

            ; O: // <- okay
            ; fmov s4, #1.0
            ; dup v4.s2, v4.s[0]
            ; fdiv V(reg(out_reg)).s2, v4.s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            ; E: // <- end
        );
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne >U // -> upper_lz

            ; tst x15, #0x1
            ; b.ne >L // -> lower_lz

            // Happy path
            ; fsqrt V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; b >E // -> end

            ; L: // <- lower_lz
            ; mov v4.s[0], V(reg(lhs_reg)).s[1]
            ; fsqrt s4, s4
            ; movi D(reg(out_reg)), #0
            ; mov V(reg(out_reg)).s[1], v4.s[0]
            ; b >E

            ; U:
            ; movz w9, #(nan_u32 >> 16), lsl 16
            ; movk w9, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w9

            ; E:
        );
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, #0.0
            ; fmov x15, d4
            ; fmul V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper <= 0.0
            ; tst x15, #0x1_0000_0000
            ; b.ne >S // -> swap

            // Test whether lhs.lower <= 0.0
            ; tst x15, #0x1
            ; b.eq >E // -> end

            // If the input interval straddles 0, then the
            // output is [0, max(lower**2, upper**2)]
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; movi D(reg(out_reg)), #0
            ; mov V(reg(out_reg)).s[1], v4.s[0]
            ; b >E // -> end

            ; S: // <- swap
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            ; E: // <- end
        );
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; fadd V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            ; rev64 v4.s2, V(reg(rhs_reg)).s2
            ; fsub V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, v4.s2
        )
    }
    fn build_sub_reg_imm(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        dynasm!(self.0
            ; fsub V(reg(out_reg)).s2, V(reg(arg)).s2, V(reg(imm)).s2
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            // Set up v4 to contain
            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
            // and v5 to contain
            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
            //
            // Multiplying them out will hit all four possible
            // combinations; then we extract the min and max
            // with vector-reducing operations
            ; rev64 v4.s2, V(reg(lhs_reg)).s2
            ; mov v4.d[1], V(reg(lhs_reg)).d[0]
            ; dup v5.d2, V(reg(rhs_reg)).d[0]

            ; fmul v4.s4, v4.s4, v5.s4
            ; fminnmv S(reg(out_reg)), v4.s4
            ; fmaxnmv s5, v4.s4
            ; mov V(reg(out_reg)).s[1], v5.s[0]
        )
    }

    fn build_mul_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let rhs_reg = self.load_imm(imm);
        dynasm!(self.0
            ; fmul V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
        );
        if imm < 0.0 {
            dynasm!(self.0
                ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2
            );
        }
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        let nan_u32 = f32::NAN.to_bits();
        dynasm!(self.0
            // Store rhs.lower > 0.0 in x15, then check rhs.lower > 0
            ; fcmp S(reg(rhs_reg)), #0.0
            ; b.gt >H // -> happy

            // Store rhs.upper < 0.0 in x15, then check rhs.upper < 0
            ; mov s4, V(reg(rhs_reg)).s[1]
            ; fcmp s4, #0.0
            ; b.lt >H

            // Sad path: rhs spans 0, so the output includes NaN
            ; movz w9, #(nan_u32 >> 16), lsl 16
            ; movk w9, #(nan_u32)
            ; dup V(reg(out_reg)).s2, w9
            ; b >E // -> end

            ; H: // >happy:
            // Set up v4 to contain
            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
            // and v5 to contain
            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
            //
            // Dividing them out will hit all four possible
            // combinations; then we extract the min and max
            // with vector-reducing operations
            ; rev64 v4.s2, V(reg(lhs_reg)).s2
            ; mov v4.d[1], V(reg(lhs_reg)).d[0]
            ; dup v5.d2, V(reg(rhs_reg)).d[0]

            ; fdiv v4.s4, v4.s4, v5.s4
            ; fminnmv S(reg(out_reg)), v4.s4
            ; fmaxnmv s5, v4.s4
            ; mov V(reg(out_reg)).s[1], v5.s[0]

            ; E: // >end
        );
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            // Basically the same as MinRegReg
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne >L // -> lhs

            ; tst x15, #0x1
            ; b.eq >B // -> both

            ; R: // LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; b >E // -> end

            ; L: // <- lhs (when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; b >E // -> end

            ; B: // <- both
            ; fmax V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2

            ; E: // <- end
        );
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0
            //  if lhs.upper < rhs.lower
            //      out = lhs
            //  elif rhs.upper < lhs.lower
            //      out = rhs
            //  else
            //      out = fmin(lhs, rhs)

            // v4 = [lhs.upper, rhs.upper]
            // v5 = [rhs.lower, lhs.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2

            // v5 = [rhs.lower > lhs.upper, lhs.lower > rhs.upper]
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne >R // -> rhs

            ; tst x15, #0x1
            ; b.eq >B // -> both

            ; L: // Fallthrough: LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; b >E // -> end

            ; R: // <- rhs (for when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; b >E

            ; B: // <- both
            ; fmin V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2

            // <- end
            ; E:
        );
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0
            ; movz w15, #(imm_u32 >> 16), lsl 16
            ; movk w15, #(imm_u32)
            ; dup V(IMM_REG as u32).s2, w15
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
            ; fmov D(reg(inout_reg)), D(reg(arg_reg))
        );
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            ; b >E

            ; V: // There was a previous value, so we have to do the comparison
            // v4 = [inout_reg.upper, arg_reg.upper]
            // v5 = [arg_reg.lower, inout_reg.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(reg(inout_reg)).s2, V(reg(arg_reg)).s2
            ; zip1 v5.s2, V(reg(arg_reg)).s2, V(reg(inout_reg)).s2

            // v5.0 = arg_reg.lower   > inout_reg.upper
            // v5.1 = inout_reg.lower > arg_reg.upper
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne >A // -> arg_reg

            ; tst x15, #0x1
            ; b.eq >B // -> both

            ; I: // Fallthrough: inout_reg < arg_reg
            // Nothing to do here, the value is already in memory so we write
            // the simplify bit then break to the exit
            ; mov w15, #1
            ; str w15, [x3]
            ; b >E // -> end
            //////////////////////////////////////////////////////////////////

            // TODO could we reorder these to consolidate common code
            // (everything except the fmov)?
            ; A: // arg_reg < inout_reg
            ; mov w15, #1 // write simplify bit
            ; str w15, [x3]
            ; fmov D(reg(inout_reg)), D(reg(arg_reg)) // copy the reg
        );
        // Set the choice exclusively
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            ; b >E // end of arg_reg < inout_reg
            //////////////////////////////////////////////////////////////////

            ; B: // ambiguous, so set choice non-exclusively
            ; fmin V(reg(inout_reg)).s2, V(reg(inout_reg)).s2, V(reg(arg_reg)).s2
        );
        set_choice_bit(self.0, choice); // non-exclusive
        dynasm!(self.0
            // end of ambiguous case (B label); fallthrough to end
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
            ; tst w15, #1
            ; b.ne >V

            // Fallthrough: there was no value, so we set it here
            // Copy the value, then branch to the end
            ; fmov D(reg(inout_reg)), D(reg(arg_reg))
        );
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            ; b >E

            ; V: // There was a previous value, so we have to do the comparison
            // v4 = [inout_reg.upper, arg_reg.upper]
            // v5 = [arg_reg.lower,   inout_reg.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(reg(inout_reg)).s2, V(reg(arg_reg)).s2
            ; zip1 v5.s2, V(reg(arg_reg)).s2, V(reg(inout_reg)).s2

            // v5.0 = arg_reg.lower > inout_reg.upper
            // v5.1 = inout_reg.lower > arg_reg.upper
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne >I // -> inout_reg

            ; tst x15, #0x1
            ; b.eq >B // -> both

            // Fallthrough: arg_reg > inout_reg
            ; A: // arg_reg > inout_reg
            ; mov w15, #1 // write simplify bit
            ; str w15, [x3]
            ; fmov D(reg(inout_reg)), D(reg(arg_reg)) // copy the reg
        );
        // Set the choice exclusively
        set_choice_exclusive(self.0, choice);
        dynasm!(self.0
            ; b >E // end of arg_reg > inout_reg
            //////////////////////////////////////////////////////////////////

            ; I: // inout_reg > arg_reg
            // Nothing to do here, the value is already in the register so we
            // write the simplify bit then break to the exit
            ; mov w15, #1
            ; str w15, [x3]
            ; b >E
            //////////////////////////////////////////////////////////////////

            ; B: // ambiguous, so set choice non-exclusively
            ; fmax V(reg(inout_reg)).s2, V(reg(inout_reg)).s2, V(reg(arg_reg)).s2
        );
        set_choice_bit(self.0, choice); // non-exclusive
        dynasm!(self.0
            // end of ambiguous case (B label)
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
        let lhs = SCRATCH_REG.wrapping_sub(OFFSET);
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
            ; fmov D(reg(out)), D(reg(arg))
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
            ; mov w15, #3
            ; strb w15, [x2, #i]
        );
        self.build_store(out, arg);
    }
}
