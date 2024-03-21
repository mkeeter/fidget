use crate::{
    eval::types::Interval,
    jit::{
        interval::IntervalAssembler, mmap::Mmap, reg, Assembler, AssemblerData,
        CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT, IMM_REG, OFFSET,
        REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi};

/// Implementation for the interval assembler on `aarch64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register   | Type                    |
/// |------------|------------|-------------------------|
/// | X          | `(s0, s1)` | `(f32, f32)`            |
/// | Y          | `(s2, s3)` | `(f32, f32)`            |
/// | Z          | `(s4, s5)` | `(f32, f32)`            |
/// | `vars`     | `x0`       | `*const f32` (array)    |
/// | `choices`  | `x1`       | `*mut u8` (array)       |
/// | `simplify` | `x2`       | `*mut u8` (single)      |
///
/// During evaluation, `x0`, `x1`, and `x2` maintain their meaning, while X, Y,
/// and Z are stored in `V0-2.S2`.  Intervals are stored in the lower two float
/// of a SIMD register `Vx` `s[0]` is the lower bound of the interval and `s[1]`
/// is the upper bound; for example, `V0.S0` represents the lower bound for X.
///
/// Here is the full table of registers used during evaluation:
///
/// | Register | Description                                          |
/// |----------|------------------------------------------------------|
/// | `v0.s2`  | X interval                                           |
/// | `v1.s2`  | Y interval                                           |
/// | `v2.s2`  | Z interval                                           |
/// | `v3.s2`  | Immediate value (`IMM_REG`)                          |
/// | `v4.s2`  | Scratch register                                     |
/// | `v5.s2`  | Scratch register                                     |
/// | `v8-15`  | Tape values (callee-saved)                           |
/// | `v16-31` | Tape values (caller-saved)                           |
/// | `x0`     | Function pointer for calls                           |
/// | `w9`     | Staging for loading immediate                        |
/// | `w14`    | Scratch space used for choice value                  |
/// | `x15`    | Miscellaneous scratch space                          |
///
/// The stack is configured as follows (representing intervals as `dX`, since
/// that's the right width to load / store):
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|------------------------------------------------------------|
/// | 0x100    | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
/// | 0xf8     | `x22`        | During functions calls, we use these        |
/// | 0xf0     | `x21`        | as temporary storage so must preserve their |
/// | 0xe8     | `x20`        | previous values on the stack                |
/// |----------|------------------------------------------------------------|
/// | 0xe0     | `d2`         | During functions calls, X/Y/Z are saved on  |
/// | 0xd8     | `d1`         | the stack                                   |
/// | 0xd0     | `d0`         |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0xc8     | `d31`        | During functions calls, caller-saved tape   |
/// | 0xc0     | `d30`        | registers are saved on the stack            |
/// | 0xb8     | `d29`        |                                             |
/// | 0xb0     | `d28`        |                                             |
/// | 0xa8     | `d27`        |                                             |
/// | 0xa0     | `d26`        |                                             |
/// | 0x98     | `d25`        |                                             |
/// | 0x90     | `d24`        |                                             |
/// | 0x88     | `d23`        |                                             |
/// | 0x80     | `d22`        |                                             |
/// | 0x78     | `d21`        |                                             |
/// | 0x70     | `d20`        |                                             |
/// | 0x68     | `d19`        |                                             |
/// | 0x60     | `d18`        |                                             |
/// | 0x58     | `d17`        |                                             |
/// | 0x50     | `d16`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x48     | `d15`        | Callee-saved registers                      |
/// | 0x40     | `d14`        |                                             |
/// | 0x38     | `d13`        |                                             |
/// | 0x30     | `d12`        |                                             |
/// | 0x28     | `d11`        |                                             |
/// | 0x20     | `d10`        |                                             |
/// | 0x18     | `d9`         |                                             |
/// | 0x10     | `d8`         |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x8      | `sp` (`x30`) | Stack frame                                 |
/// | 0x0      | `fp` (`x29`) | [current value for sp]                      |
/// ```
const STACK_SIZE: u32 = 0x100;

impl Assembler for IntervalAssembler {
    type Data = Interval;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        out.prepare_stack(slot_count, STACK_SIZE as usize);
        dynasm!(out.ops
            // Preserve frame and link register, and set up the frame pointer
            ; stp   x29, x30, [sp, 0x0]
            ; mov   x29, sp

            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, 0x10]
            ; stp   d10, d11, [sp, 0x20]
            ; stp   d12, d13, [sp, 0x30]
            ; stp   d14, d15, [sp, 0x40]

            // Arguments are passed in S0-5; collect them into V0-1
            ; mov v0.s[1], v1.s[0]
            ; mov v1.s[0], v2.s[0]
            ; mov v1.s[1], v3.s[0]
            ; mov v2.s[0], v4.s[0]
            ; mov v2.s[1], v5.s[0]
        );
        Self(out)
    }

    fn bytes_per_clause() -> usize {
        40
    }

    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem) + STACK_SIZE;
        assert!(sp_offset <= 32768);
        dynasm!(self.0.ops ; ldr D(reg(dst_reg)), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem) + STACK_SIZE;
        assert!(sp_offset <= 32768);
        dynasm!(self.0.ops ; str D(reg(src_reg)), [sp, #(sp_offset)])
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; fmov D(reg(out_reg)), D(src_arg as u32));
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0.ops
            ; ldr w15, [x0, #(src_arg * 4)]
            ; dup V(reg(out_reg)).s2, w15
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn interval_sin(v: Interval) -> Interval {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, interval_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_cos(f: Interval) -> Interval {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_tan(f: Interval) -> Interval {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_asin(f: Interval) -> Interval {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_acos(f: Interval) -> Interval {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_atan(f: Interval) -> Interval {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_exp(f: Interval) -> Interval {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "C" fn float_ln(f: Interval) -> Interval {
            f.ln()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ln);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops ; fmov D(reg(out_reg)), D(reg(lhs_reg)))
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; fneg V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2
        )
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store lhs < 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, 0.0
            ; fmov x15, d4

            // Store abs(lhs) in V(reg(out_reg))
            ; fabs V(reg(out_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper < 0
            ; tst x15, 0x1_0000_0000
            ; b.ne 24 // -> upper_lz

            // Check whether lhs.lower < 0
            ; tst x15, 0x1

            // otherwise, we're good; return the original
            ; b.eq 20 // -> end

            // if lhs.lower < 0, then the output is
            //  [0.0, max(abs(lower, upper))]
            ; movi d4, 0
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; fmov D(reg(out_reg)), d4
            // Fall through to do the swap

            // <- upper_lz
            // if upper < 0
            //   return [-upper, -lower]
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Check whether lhs.lower > 0.0
            ; fcmp S(reg(lhs_reg)), 0.0
            ; b.gt 28 // -> okay

            // Check whether lhs.upper < 0.0
            ; mov s4, V(reg(lhs_reg)).s[1]
            ; fcmp s4, 0.0
            ; b.mi 16 // -> okay

            // Bad case: the division spans 0, so return NaN
            ; mov w15, f32::NAN.to_bits().into()
            ; dup V(reg(out_reg)).s2, w15
            ; b 20 // -> end

            // <- okay
            ; fmov s4, 1.0
            ; dup v4.s2, v4.s[0]
            ; fdiv V(reg(out_reg)).s2, v4.s2, V(reg(lhs_reg)).s2
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store lhs < 0.0 in x15
            ; fcmlt v4.s2, V(reg(lhs_reg)).s2, 0.0
            ; fmov x15, d4

            ; tst x15, 0x1
            ; b.ne 12 // -> lower_lz

            // Happy path
            ; fsqrt V(reg(out_reg)).s2, V(reg(lhs_reg)).s2
            ; b 12 // -> end

            // <- upper_lz
            ; mov w9, f32::NAN.to_bits().into()
            ; dup V(reg(out_reg)).s2, w9

            // <- end
        )
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(reg(lhs_reg)).s2, 0.0
            ; fmov x15, d4
            ; fmul V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(lhs_reg)).s2

            // Check whether lhs.upper <= 0.0
            ; tst x15, 0x1_0000_0000
            ; b.ne 28 // -> swap

            // Test whether lhs.lower <= 0.0
            ; tst x15, 0x1
            ; b.eq 24 // -> end

            // If the input interval straddles 0, then the
            // output is [0, max(lower**2, upper**2)]
            ; fmaxnmv s4, V(reg(out_reg)).s4
            ; movi D(reg(out_reg)), 0
            ; mov V(reg(out_reg)).s[1], v4.s[0]
            ; b 8 // -> end

            // <- swap
            ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2

            // <- end
        )
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fadd V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; rev64 v4.s2, V(reg(rhs_reg)).s2
            ; fsub V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, v4.s2
        )
    }
    fn build_sub_reg_imm(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        dynasm!(self.0.ops
            ; fsub V(reg(out_reg)).s2, V(reg(arg)).s2, V(reg(imm)).s2
        )
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
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
        dynasm!(self.0.ops
            ; fmul V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
        );
        if imm < 0.0 {
            dynasm!(self.0.ops
                ; rev64 V(reg(out_reg)).s2, V(reg(out_reg)).s2
            );
        }
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Store rhs.lower > 0.0 in x15, then check rhs.lower > 0
            ; fcmp S(reg(rhs_reg)), 0.0
            ; b.gt 28 // -> happy

            // Store rhs.upper < 0.0 in x15, then check rhs.upper < 0
            ; mov s4, V(reg(rhs_reg)).s[1]
            ; fcmp s4, 0.0
            ; b.lt 16

            // Sad path: rhs spans 0, so the output includes NaN
            ; mov w9, f32::NAN.to_bits().into()
            ; dup V(reg(out_reg)).s2, w9
            ; b 32 // -> end

            // >happy:
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

            // >end
        )
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Basically the same as MinRegReg
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5
            ; ldrb w14, [x1]

            ; tst x15, 0x1_0000_0000
            ; b.ne 28 // -> lhs

            ; tst x15, 0x1
            ; b.eq 36 // -> both

            // LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 28 // -> end

            // <- lhs (when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 12 // -> end

            // <- both
            ; fmax V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; orr w14, w14, #CHOICE_BOTH

            // <- end
            ; strb w14, [x1], 1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            //  if lhs.upper < rhs.lower
            //      *choices++ |= CHOICE_LEFT
            //      out = lhs
            //  elif rhs.upper < lhs.lower
            //      *choices++ |= CHOICE_RIGHT
            //      out = rhs
            //  else
            //      *choices++ |= CHOICE_BOTH
            //      out = fmin(lhs, rhs)

            // v4 = [lhs.upper, rhs.upper]
            // v5 = [rhs.lower, lhs.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; zip1 v5.s2, V(reg(rhs_reg)).s2, V(reg(lhs_reg)).s2

            // v5 = [rhs.lower > lhs.upper, lhs.lower > rhs.upper]
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5
            ; ldrb w14, [x1]

            ; tst x15, 0x1_0000_0000
            ; b.ne 28 // -> rhs

            ; tst x15, 0x1
            ; b.eq 36 // -> both

            // Fallthrough: LHS < RHS
            ; fmov D(reg(out_reg)), D(reg(lhs_reg))
            ; orr w14, w14, #CHOICE_LEFT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 28 // -> end

            // <- rhs (for when RHS < LHS)
            ; fmov D(reg(out_reg)), D(reg(rhs_reg))
            ; orr w14, w14, #CHOICE_RIGHT
            ; strb w14, [x2, 0] // write a non-zero value to simplify
            ; b 12

            // <- both
            ; fmin V(reg(out_reg)).s2, V(reg(lhs_reg)).s2, V(reg(rhs_reg)).s2
            ; orr w14, w14, #CHOICE_BOTH

            // <- end
            ; strb w14, [x1], 1 // post-increment
        )
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; movz w15, #(imm_u32 >> 16), lsl 16
            ; movk w15, #(imm_u32)
            ; dup V(IMM_REG as u32).s2, w15
        );
        IMM_REG.wrapping_sub(OFFSET)
    }

    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        assert!(self.0.mem_offset < 4096);
        if self.0.saved_callee_regs {
            dynasm!(self.0.ops
                // Restore callee-saved registers
                ; ldp x20, x21, [sp, 0xe8]
                ; ldr x22, [sp, 0xf8]
            )
        }
        dynasm!(self.0.ops
            // Prepare our return value
            ; mov  s0, V(reg(out_reg)).s[0]
            ; mov  s1, V(reg(out_reg)).s[1]

            // Restore frame and link register
            ; ldp   x29, x30, [sp, 0x0]

            // Restore callee-saved floating-point registers
            ; ldp   d8, d9, [sp, 0x10]
            ; ldp   d10, d11, [sp, 0x20]
            ; ldp   d12, d13, [sp, 0x30]
            ; ldp   d14, d15, [sp, 0x40]

            // Fix up the stack
            ; add sp, sp, #(self.0.mem_offset as u32)

            ; ret
        );

        self.0.ops.finalize()
    }
}

impl IntervalAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "C" fn(Interval) -> Interval,
    ) {
        if !self.0.saved_callee_regs {
            dynasm!(self.0.ops
                // Back up a few callee-saved registers that we're about to use
                ; stp x20, x21, [sp, 0xe8]
                ; str x22, [sp, 0xf8]
            );
            self.0.saved_callee_regs = true;
        }

        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up our current state to callee-saved registers
            ; mov x20, x0
            ; mov x21, x1
            ; mov x22, x2

            // Back up our state
            ; stp d16, d17, [sp, 0x50]
            ; stp d18, d19, [sp, 0x60]
            ; stp d20, d21, [sp, 0x70]
            ; stp d22, d23, [sp, 0x80]
            ; stp d24, d25, [sp, 0x90]
            ; stp d26, d27, [sp, 0xa0]
            ; stp d28, d29, [sp, 0xb0]
            ; stp d30, d31, [sp, 0xc0]
            ; stp d0, d1, [sp, 0xd0]
            ; str d2, [sp, 0xe0]

            // Load the function address, awkwardly, into a caller-saved
            // register (so we only need to do this once)
            ; movz x0, #((addr >> 48) as u32), lsl 48
            ; movk x0, #((addr >> 32) as u32), lsl 32
            ; movk x0, #((addr >> 16) as u32), lsl 16
            ; movk x0, #(addr as u32)

            // Prepare to call our stuff!
            ; mov s0, V(reg(arg_reg)).s[0]
            ; mov s1, V(reg(arg_reg)).s[1]

            ; blr x0

            // Restore floating-point state
            ; ldp d16, d17, [sp, 0x50]
            ; ldp d18, d19, [sp, 0x60]
            ; ldp d20, d21, [sp, 0x70]
            ; ldp d22, d23, [sp, 0x80]
            ; ldp d24, d25, [sp, 0x90]
            ; ldp d26, d27, [sp, 0xa0]
            ; ldp d28, d29, [sp, 0xb0]
            ; ldp d30, d31, [sp, 0xc0]

            // Set our output value
            ; mov V(reg(out_reg)).s[0], v0.s[0]
            ; mov V(reg(out_reg)).s[1], v1.s[0]

            // Restore X/Y/Z values
            ; ldp d0, d1, [sp, 0xd0]
            ; ldr d2, [sp, 0xe0]

            // Restore registers
            ; mov x0, x20
            ; mov x1, x21
            ; mov x2, x22
        );
    }
}
