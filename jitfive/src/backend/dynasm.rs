use dynasmrt::{
    aarch64::Assembler, dynasm, AssemblyOffset, DynasmApi, DynasmLabelApi,
    ExecutableBuffer,
};
use num_traits::FromPrimitive;

use crate::backend::{
    common::Choice,
    tape32::{ClauseOp32, ClauseOp64, Tape},
};

/// We can use registers v8-v15 (callee saved) and v16-v31 (caller saved)
pub const REGISTER_LIMIT: usize = 24;
const REG_OFFSET: u32 = 8;

const CHOICE_LEFT: u64 = Choice::Left as u64;
const CHOICE_RIGHT: u64 = Choice::Right as u64;
const CHOICE_BOTH: u64 = Choice::Both as u64;

struct FloatAssembler {
    ops: Assembler,
    shape_fn: AssemblyOffset,
    reg_count: usize,
    stack_space: u32,
}

impl FloatAssembler {
    fn init(reg_count: usize, total_slots: usize) -> Self {
        assert!(reg_count <= REGISTER_LIMIT);

        let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
        dynasm!(ops
            ; -> shape_fn:
        );
        let shape_fn = ops.offset();

        let stack_space = total_slots.saturating_sub(reg_count) as u32 * 4;
        // Ensure alignment
        let stack_space = ((stack_space + 15) / 16) * 16;

        dynasm!(ops
            // Preserve frame and link register
            ; stp   x29, x30, [sp, #-16]!
            // Preserve sp
            ; mov   x29, sp
            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, #-16]!
            ; stp   d10, d11, [sp, #-16]!
            ; stp   d12, d13, [sp, #-16]!
            ; stp   d14, d15, [sp, #-16]!
            ; sub   sp, sp, #(stack_space)
        );

        Self {
            ops,
            shape_fn,
            reg_count,
            stack_space,
        }
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u32, src_mem: u32) {
        let sp_offset = 4 * (src_mem - self.reg_count as u32);
        dynasm!(self.ops ; ldr S(dst_reg), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u32) {
        let sp_offset = 4 * (dst_mem - self.reg_count as u32);
        dynasm!(self.ops ; str S(src_reg), [sp, #(sp_offset)])
    }
    /// Swaps a register and memory location, using S4 as an imtermediary
    fn build_swap(&mut self, reg: u32, mem: u32) {
        let sp_offset = 4 * (mem - self.reg_count as u32);
        dynasm!(self.ops
            ; fmov s4, S(reg)
            ; ldr S(reg), [sp, #(sp_offset)]
            ; str s4, [sp, #(sp_offset)]
        );
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u32, src_arg: u32) {
        dynasm!(self.ops ; fmov S(out_reg), S(src_arg));
    }
    fn build_copy(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fmov S(out_reg), S(lhs_reg))
    }
    fn build_neg(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fneg S(out_reg), S(lhs_reg))
    }
    fn build_abs(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fabs S(out_reg), S(lhs_reg))
    }
    fn build_recip(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            ; fmov s7, #1.0
            ; fdiv S(out_reg), s7, S(lhs_reg)
        )
    }
    fn build_sqrt(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fsqrt S(out_reg), S(lhs_reg))
    }
    fn build_square(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fmul S(out_reg), S(lhs_reg), S(lhs_reg))
    }
    fn build_add(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fadd S(out_reg), S(lhs_reg), S(rhs_reg))
    }
    fn build_sub(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fsub S(out_reg), S(lhs_reg), S(rhs_reg))
    }
    fn build_mul(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fmul S(out_reg), S(lhs_reg), S(rhs_reg))
    }
    fn build_max(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fmax S(out_reg), S(lhs_reg), S(rhs_reg))
    }
    fn build_min(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fmin S(out_reg), S(lhs_reg), S(rhs_reg))
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) {
        let imm_u32 = imm.to_bits();
        dynasm!(self.ops
            ; movz w9, #(imm_u32 >> 16), lsl 16
            ; movk w9, #(imm_u32)
            ; fmov s4, w9
        );
    }

    fn finalize(mut self, out_reg: u32) -> (ExecutableBuffer, AssemblyOffset) {
        dynasm!(self.ops
            // Prepare our return value
            ; fmov  s0, S(out_reg)
            // Restore stack space used for spills
            ; add   sp, sp, #(self.stack_space)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        (self.ops.finalize().unwrap(), self.shape_fn)
    }
}

pub fn build_float_fn(t: &Tape) -> FloatFuncHandle {
    assert!(t.reg_count <= REGISTER_LIMIT);

    let mut asm = FloatAssembler::init(t.reg_count, t.total_slots);

    let mut iter = t.tape.iter().rev();
    while let Some(v) = iter.next() {
        if v & (1 << 30) == 0 {
            let op = (v >> 24) & ((1 << 6) - 1);
            let op = ClauseOp32::from_u32(op).unwrap();
            match op {
                ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                    let reg = (v & 0xFF) + REG_OFFSET;
                    let mem = (v >> 8) & 0xFFFF;
                    match op {
                        ClauseOp32::Load => asm.build_load(reg, mem),
                        ClauseOp32::Store => asm.build_store(mem, reg),
                        ClauseOp32::Swap => asm.build_swap(reg, mem),
                        _ => unreachable!(),
                    }
                }
                ClauseOp32::Input => {
                    let input = (v >> 16) & 0xFF;
                    assert!(input < 3);
                    let out_reg = (v & 0xFF) + REG_OFFSET;
                    asm.build_input(out_reg, input)
                }
                ClauseOp32::CopyReg
                | ClauseOp32::NegReg
                | ClauseOp32::AbsReg
                | ClauseOp32::RecipReg
                | ClauseOp32::SqrtReg
                | ClauseOp32::SquareReg => {
                    let lhs = ((v >> 16) & 0xFF) + REG_OFFSET;
                    let out = (v & 0xFF) + REG_OFFSET;
                    match op {
                        ClauseOp32::CopyReg => asm.build_copy(out, lhs),
                        ClauseOp32::NegReg => asm.build_neg(out, lhs),
                        ClauseOp32::AbsReg => asm.build_abs(out, lhs),
                        ClauseOp32::RecipReg => asm.build_recip(out, lhs),
                        ClauseOp32::SqrtReg => asm.build_sqrt(out, lhs),
                        ClauseOp32::SquareReg => asm.build_square(out, lhs),
                        _ => unreachable!(),
                    };
                }

                ClauseOp32::AddRegReg
                | ClauseOp32::MulRegReg
                | ClauseOp32::SubRegReg
                | ClauseOp32::MinRegReg
                | ClauseOp32::MaxRegReg => {
                    let lhs = ((v >> 16) & 0xFF) + REG_OFFSET;
                    let rhs = ((v >> 8) & 0xFF) + REG_OFFSET;
                    let out = (v & 0xFF) + REG_OFFSET;
                    match op {
                        ClauseOp32::AddRegReg => asm.build_add(out, lhs, rhs),
                        ClauseOp32::MulRegReg => asm.build_mul(out, lhs, rhs),
                        ClauseOp32::SubRegReg => asm.build_sub(out, lhs, rhs),
                        ClauseOp32::MinRegReg => asm.build_min(out, lhs, rhs),
                        ClauseOp32::MaxRegReg => asm.build_max(out, lhs, rhs),
                        ClauseOp32::Load | ClauseOp32::Store => {
                            unreachable!()
                        }
                        _ => unreachable!(),
                    };
                }
            }
        } else {
            let op = (v >> 16) & ((1 << 14) - 1);
            let op = ClauseOp64::from_u32(op).unwrap();
            let arg = ((v >> 8) & 0xFF) + REG_OFFSET;
            let out = (v & 0xFF) + REG_OFFSET;

            let next = iter.next().unwrap();
            asm.load_imm(f32::from_bits(*next));

            match op {
                ClauseOp64::AddRegImm => asm.build_add(out, arg, 4),
                ClauseOp64::MulRegImm => asm.build_mul(out, arg, 4),
                ClauseOp64::SubImmReg => asm.build_sub(out, 4, arg),
                ClauseOp64::SubRegImm => asm.build_sub(out, arg, 4),
                ClauseOp64::MinRegImm => asm.build_min(out, arg, 4),
                ClauseOp64::MaxRegImm => asm.build_max(out, arg, 4),
                ClauseOp64::CopyImm => asm.build_copy(out, 4),
            };
        }
    }

    let (buf, shape_fn) = asm.finalize(REG_OFFSET);
    let fn_pointer = buf.ptr(shape_fn);
    FloatFuncHandle {
        _buf: buf,
        fn_pointer,
    }
}

////////////////////////////////////////////////////////////////////////////////

struct IntervalAssembler {
    ops: Assembler,
    shape_fn: AssemblyOffset,
    reg_count: usize,
    stack_space: u32,
}

impl IntervalAssembler {
    fn init(reg_count: usize, total_slots: usize) -> Self {
        assert!(reg_count <= REGISTER_LIMIT);

        let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
        dynasm!(ops
            ; -> shape_fn:
        );
        let shape_fn = ops.offset();

        let stack_space = total_slots.saturating_sub(reg_count) as u32 * 4 * 2;
        // Ensure alignment
        let stack_space = ((stack_space + 15) / 16) * 16;

        dynasm!(ops
            // Preserve frame and link register
            ; stp   x29, x30, [sp, #-16]!
            // Preserve sp
            ; mov   x29, sp
            // Preserve callee-saved floating-point registers
            ; stp   d8, d9, [sp, #-16]!
            ; stp   d10, d11, [sp, #-16]!
            ; stp   d12, d13, [sp, #-16]!
            ; stp   d14, d15, [sp, #-16]!
            ; sub   sp, sp, #(stack_space)

            // Arguments are passed in S0-5; collect them into V0-1
            ; mov v0.s[1], v1.s[0]
            ; mov v1.s[0], v2.s[0]
            ; mov v1.s[1], v3.s[0]
            ; mov v2.s[0], v4.s[0]
            ; mov v2.s[1], v5.s[0]
        );

        Self {
            ops,
            shape_fn,
            reg_count,
            stack_space,
        }
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u32, src_mem: u32) {
        let sp_offset = 2 * 4 * (src_mem - self.reg_count as u32);
        dynasm!(self.ops ; ldr D(dst_reg), [sp, #(sp_offset)])
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u32) {
        let sp_offset = 2 * 4 * (dst_mem - self.reg_count as u32);
        dynasm!(self.ops ; str D(src_reg), [sp, #(sp_offset)])
    }
    /// Swaps a register and memory location, using S4 as an imtermediary
    fn build_swap(&mut self, reg: u32, mem: u32) {
        let sp_offset = 2 * 4 * (mem - self.reg_count as u32);
        dynasm!(self.ops
            ; fmov d4, D(reg)
            ; ldr D(reg), [sp, #(sp_offset)]
            ; str d4, [sp, #(sp_offset)]
        );
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u32, src_arg: u32) {
        dynasm!(self.ops ; fmov D(out_reg), D(src_arg));
    }
    fn build_copy(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops ; fmov D(out_reg), D(lhs_reg))
    }
    fn build_neg(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            ; fneg V(out_reg).s2, V(lhs_reg).s2
            ; rev64 V(out_reg).s2, V(out_reg).s2
        )
    }
    fn build_abs(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            // Store lhs < 0.0 in x15
            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
            ; fmov x15, d4

            // Store abs(lhs) in V(out_reg)
            ; fabs V(out_reg).s2, V(lhs_reg).s2

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> upper_lz

            // Check whether lhs.lower < 0
            ; tst x15, #0x1

            // otherwise, we're good; return the original
            ; b.eq #20 // -> end

            // if lhs.lower < 0, then the output is
            //  [0.0, max(abs(lower, upper))]
            ; movi d4, #0
            ; fmaxnmv s4, V(out_reg).s4
            ; fmov D(out_reg), d4
            // Fall through to do the swap

            // <- upper_lz
            // if upper < 0
            //   return [-upper, -lower]
            ; rev64 V(out_reg).s2, V(out_reg).s2

            // <- end
        )
    }
    fn build_recip(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            // Check whether lhs.lower > 0.0
            ; fcmgt s4, S(lhs_reg), 0.0
            ; fmov w15, s4
            ; tst w15, #0x1
            ; b.ne #40 // -> okay

            // Check whether lhs.upper < 0.0
            ; mov s4, V(lhs_reg).s[1]
            ; fcmlt s4, s4, 0.0
            ; fmov w15, s4
            ; tst w15, #0x1
            ; b.ne #20 // -> okay

            // Bad case: the division spans 0, so return NaN
            ; movz w15, #(nan_u32 >> 16), lsl 16
            ; movk w15, #(nan_u32)
            ; dup V(out_reg).s2, w15
            ; b #20 // -> end

            // <- okay
            ; fmov s4, #1.0
            ; dup v4.s2, v4.s[0]
            ; fdiv V(out_reg).s2, v4.s2, V(lhs_reg).s2
            ; rev64 V(out_reg).s2, V(out_reg).s2

            // <- end
        )
    }
    fn build_sqrt(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            // Store lhs <= 0.0 in x8
            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
            ; fmov x15, d4

            // Check whether lhs.upper < 0
            ; tst x15, #0x1_0000_0000
            ; b.ne #40 // -> upper_lz

            ; tst x15, #0x1
            ; b.ne #12 // -> lower_lz

            // Happy path
            ; fsqrt V(out_reg).s2, V(lhs_reg).s2
            ; b #36 // -> end

            // <- lower_lz
            ; mov v4.s[0], V(lhs_reg).s[1]
            ; fsqrt s4, s4
            ; movi D(out_reg), #0
            ; mov V(out_reg).s[1], v4.s[0]
            ; b #16

            // <- upper_lz
            ; movz w9, #(nan_u32 >> 16), lsl 16
            ; movk w9, #(nan_u32)
            ; dup V(out_reg).s2, w9

            // <- end
        )
    }
    fn build_square(&mut self, out_reg: u32, lhs_reg: u32) {
        dynasm!(self.ops
            // Store lhs <= 0.0 in x15
            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
            ; fmov x15, d4
            ; fmul V(out_reg).s2, V(lhs_reg).s2, V(lhs_reg).s2

            // Check whether lhs.upper <= 0.0
            ; tst x15, #0x1_0000_0000
            ; b.ne #28 // -> swap

            // Test whether lhs.lower <= 0.0
            ; tst x15, #0x1
            ; b.eq #24 // -> end

            // If the input interval straddles 0, then the
            // output is [0, max(lower**2, upper**2)]
            ; fmaxnmv s4, V(out_reg).s4
            ; movi D(out_reg), #0
            ; mov V(out_reg).s[1], v4.s[0]
            ; b #8 // -> end

            // <- swap
            ; rev64 V(out_reg).s2, V(out_reg).s2

            // <- end
        )
    }
    fn build_add(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops ; fadd V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2)
    }
    fn build_sub(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops
            ; rev64 v4.s2, V(rhs_reg).s2
            ; fsub V(out_reg).s2, V(lhs_reg).s2, v4.s2
        )
    }
    fn build_mul(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops
            // Set up v4 to contain
            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
            // and v5 to contain
            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
            //
            // Multiplying them out will hit all four possible
            // combinations; then we extract the min and max
            // with vector-reducing operations
            ; rev64 v4.s2, V(lhs_reg).s2
            ; mov v4.d[1], V(lhs_reg).d[0]
            ; dup v5.d2, V(rhs_reg).d[0]

            ; fmul v4.s4, v4.s4, v5.s4
            ; fminnmv S(out_reg), v4.s4
            ; fmaxnmv s5, v4.s4
            ; mov V(out_reg).s[1], v5.s[0]
        )
    }
    fn build_max(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops
            // Basically the same as MinRegReg
            ; zip2 v4.s2, V(lhs_reg).s2, V(rhs_reg).s2
            ; zip1 v5.s2, V(rhs_reg).s2, V(lhs_reg).s2
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> lhs

            ; tst x15, #0x1
            ; b.eq #28 // -> both

            // LHS < RHS
            ; fmov D(out_reg), D(rhs_reg)
            ; mov w16, #CHOICE_RIGHT
            ; b #24 // -> end

            // <- lhs (when RHS < LHS)
            ; fmov D(out_reg), D(lhs_reg)
            ; mov w16, #CHOICE_LEFT
            ; b #12 // -> end

            // <- both
            ; fmax V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
            ; mov w16, #CHOICE_BOTH

            // <- end
            ; strb w16, [x0], #1 // post-increment
        )
    }
    fn build_min(&mut self, out_reg: u32, lhs_reg: u32, rhs_reg: u32) {
        dynasm!(self.ops
            //  if lhs.upper < rhs.lower
            //      *choices++ = CHOICE_LEFT
            //      out = lhs
            //  elif rhs.upper < lhs.lower
            //      *choices++ = CHOICE_RIGHT
            //      out = rhs
            //  else
            //      *choices++ = CHOICE_BOTH
            //      out = fmin(lhs, rhs)

            // v4 = [lhs.upper, rhs.upper]
            // v5 = [rhs.lower, lhs.lower]
            // This lets us do two comparisons simultaneously
            ; zip2 v4.s2, V(lhs_reg).s2, V(rhs_reg).s2
            ; zip1 v5.s2, V(rhs_reg).s2, V(lhs_reg).s2
            ; fcmgt v5.s2, v5.s2, v4.s2
            ; fmov x15, d5

            ; tst x15, #0x1_0000_0000
            ; b.ne #24 // -> rhs

            ; tst x15, #0x1
            ; b.eq #28 // -> both

            // LHS < RHS
            ; fmov D(out_reg), D(lhs_reg)
            ; mov w16, #CHOICE_LEFT
            ; b #24 // -> end

            // <- rhs (for when RHS < LHS)
            ; fmov D(out_reg), D(rhs_reg)
            ; mov w16, #CHOICE_RIGHT
            ; b #12

            // <- both
            ; fmin V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
            ; mov w16, #CHOICE_BOTH

            // <- end
            ; strb w16, [x0], #1 // post-increment
        )
    }

    /// Loads an immediate into register S4, using W9 as an intermediary
    fn load_imm(&mut self, imm: f32) {
        let imm_u32 = imm.to_bits();
        dynasm!(self.ops
            ; movz w15, #(imm_u32 >> 16), lsl 16
            ; movk w15, #(imm_u32)
            ; dup v4.s2, w15
        );
    }

    fn finalize(mut self, out_reg: u32) -> (ExecutableBuffer, AssemblyOffset) {
        dynasm!(self.ops
            // Prepare our return value
            ; mov  s0, V(REG_OFFSET).s[0]
            ; mov  s1, V(REG_OFFSET).s[1]
            // Restore stack space used for spills
            ; add   sp, sp, #(self.stack_space)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret
        );

        (self.ops.finalize().unwrap(), self.shape_fn)
    }
}

/// Alright, here's the plan.
///
/// We're calling a function of the form
/// ```
/// # type IntervalFn =
/// extern "C" fn([f32; 2], [f32; 2], [f32; 2], *mut u8) -> [f32; 2];
/// ```
///
/// The first three arguments are `x`, `y`, and `z` intervals.  They come packed
/// into `s0-5`, and we shuffle them into SIMD registers `V0.2S`, `V1.2S`, and
/// `V2.2s` respectively.
///
/// The last argument is a pointer to the `choices` array, which is populated
/// by `min` and `max` opcodes.  It comes in the `x0` register, which is
/// unchanged by our function.
///
/// During evaluation, each SIMD register stores an interval.  `s[0]` is the
/// lower bound of the interval and `s[1]` is the upper bound.
///
/// The input tape must be planned with a <= 24 register limit.  We use hardware
/// `V8.2S` through `V32.2S` to store our tape registers, and put everything
/// else on the stack.
///
/// `V4.2S` through `V7.2S` are used for scratch values within a single opcode
/// (e.g. storing intermediate values when calculating `min` or `max`).
///
/// In general, expect to use `v4` and `v5` for intermediate (float) values,
/// and `[x,w]15` for intermediate integer values.  These are all caller-saved,
/// so we can trash them at will.
pub fn build_interval_fn(t: &Tape) -> IntervalFuncHandle {
    assert!(t.reg_count <= REGISTER_LIMIT);

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    dynasm!(ops
    ; -> shape_fn:
    );
    let shape_fn = ops.offset();

    let stack_space = t.total_slots.saturating_sub(t.reg_count) as u32 * 4 * 2;
    // Ensure alignment
    let stack_space = ((stack_space + 15) / 16) * 16;

    dynasm!(ops
        // Preserve frame and link register
        ; stp   x29, x30, [sp, #-16]!
        // Preserve sp
        ; mov   x29, sp
        // Preserve callee-saved floating-point registers
        ; stp   d8, d9, [sp, #-16]!
        ; stp   d10, d11, [sp, #-16]!
        ; stp   d12, d13, [sp, #-16]!
        ; stp   d14, d15, [sp, #-16]!
        ; sub   sp, sp, #(stack_space)

        // Arguments are passed in S0-5; collect them into V0-1
        ; mov v0.s[1], v1.s[0]
        ; mov v1.s[0], v2.s[0]
        ; mov v1.s[1], v3.s[0]
        ; mov v2.s[0], v4.s[0]
        ; mov v2.s[1], v5.s[0]
    );

    // Helper constant for when we need to inject a NaN
    let nan_u32 = f32::NAN.to_bits();

    let mut iter = t.tape.iter().rev();
    while let Some(v) = iter.next() {
        if v & (1 << 30) == 0 {
            let op = (v >> 24) & ((1 << 6) - 1);
            let op = ClauseOp32::from_u32(op).unwrap();
            match op {
                ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                    let fast_reg = (v & 0xFF) + REG_OFFSET;
                    let extended_reg = (v >> 8) & 0xFFFF;
                    let sp_offset = 2 * 4 * (extended_reg - t.reg_count as u32);
                    // We'll pretend we're reading and writing doubles (D), but
                    // are actually loading 2x floats (S).
                    match op {
                        ClauseOp32::Load => {
                            dynasm!(ops
                                ; ldr D(fast_reg), [sp, #(sp_offset)]
                            );
                        }
                        ClauseOp32::Store => {
                            dynasm!(ops
                                ; str D(fast_reg), [sp, #(sp_offset)]
                            );
                        }
                        ClauseOp32::Swap => {
                            dynasm!(ops
                                ; fmov d4, D(fast_reg)
                                ; ldr D(fast_reg), [sp, #(sp_offset)]
                                ; str d4, [sp, #(sp_offset)]
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                ClauseOp32::Input => {
                    let input = (v >> 16) & 0xFF;
                    assert!(input < 3);
                    let out_reg = (v & 0xFF) + REG_OFFSET;
                    dynasm!(ops
                        ; fmov D(out_reg), D(input)
                    );
                }
                ClauseOp32::CopyReg
                | ClauseOp32::NegReg
                | ClauseOp32::AbsReg
                | ClauseOp32::RecipReg
                | ClauseOp32::SqrtReg
                | ClauseOp32::SquareReg => {
                    let lhs_reg = ((v >> 16) & 0xFF) + REG_OFFSET;
                    let out_reg = (v & 0xFF) + REG_OFFSET;
                    match op {
                        ClauseOp32::CopyReg => dynasm!(ops
                            ; fmov D(out_reg), D(lhs_reg)
                        ),
                        ClauseOp32::NegReg => dynasm!(ops
                            ; fneg V(out_reg).s2, V(lhs_reg).s2
                            ; rev64 V(out_reg).s2, V(out_reg).s2
                        ),
                        ClauseOp32::AbsReg => dynasm!(ops
                            // Store lhs < 0.0 in x15
                            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
                            ; fmov x15, d4

                            // Store abs(lhs) in V(out_reg)
                            ; fabs V(out_reg).s2, V(lhs_reg).s2

                            // Check whether lhs.upper < 0
                            ; tst x15, #0x1_0000_0000
                            ; b.ne #24 // -> upper_lz

                            // Check whether lhs.lower < 0
                            ; tst x15, #0x1

                            // otherwise, we're good; return the original
                            ; b.eq #20 // -> end

                            // if lhs.lower < 0, then the output is
                            //  [0.0, max(abs(lower, upper))]
                            ; movi d4, #0
                            ; fmaxnmv s4, V(out_reg).s4
                            ; fmov D(out_reg), d4
                            // Fall through to do the swap

                            // <- upper_lz
                            // if upper < 0
                            //   return [-upper, -lower]
                            ; rev64 V(out_reg).s2, V(out_reg).s2

                            // <- end
                        ),
                        ClauseOp32::RecipReg => dynasm!(ops
                            // Check whether lhs.lower > 0.0
                            ; fcmgt s4, S(lhs_reg), 0.0
                            ; fmov w15, s4
                            ; tst w15, #0x1
                            ; b.ne #40 // -> okay

                            // Check whether lhs.upper < 0.0
                            ; mov s4, V(lhs_reg).s[1]
                            ; fcmlt s4, s4, 0.0
                            ; fmov w15, s4
                            ; tst w15, #0x1
                            ; b.ne #20 // -> okay

                            // Bad case: the division spans 0, so return NaN
                            ; movz w15, #(nan_u32 >> 16), lsl 16
                            ; movk w15, #(nan_u32)
                            ; dup V(out_reg).s2, w15
                            ; b #20 // -> end

                            // <- okay
                            ; fmov s4, #1.0
                            ; dup v4.s2, v4.s[0]
                            ; fdiv V(out_reg).s2, v4.s2, V(lhs_reg).s2
                            ; rev64 V(out_reg).s2, V(out_reg).s2

                            // <- end
                        ),
                        ClauseOp32::SqrtReg => dynasm!(ops
                            // Store lhs <= 0.0 in x8
                            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
                            ; fmov x15, d4

                            // Check whether lhs.upper < 0
                            ; tst x15, #0x1_0000_0000
                            ; b.ne #40 // -> upper_lz

                            ; tst x15, #0x1
                            ; b.ne #12 // -> lower_lz

                            // Happy path
                            ; fsqrt V(out_reg).s2, V(lhs_reg).s2
                            ; b #36 // -> end

                            // <- lower_lz
                            ; mov v4.s[0], V(lhs_reg).s[1]
                            ; fsqrt s4, s4
                            ; movi D(out_reg), #0
                            ; mov V(out_reg).s[1], v4.s[0]
                            ; b #16

                            // <- upper_lz
                            ; movz w9, #(nan_u32 >> 16), lsl 16
                            ; movk w9, #(nan_u32)
                            ; dup V(out_reg).s2, w9

                            // <- end
                        ),
                        ClauseOp32::SquareReg => dynasm!(ops
                            // Store lhs <= 0.0 in x15
                            ; fcmle v4.s2, V(lhs_reg).s2, #0.0
                            ; fmov x15, d4
                            ; fmul V(out_reg).s2, V(lhs_reg).s2, V(lhs_reg).s2

                            // Check whether lhs.upper <= 0.0
                            ; tst x15, #0x1_0000_0000
                            ; b.ne #28 // -> swap

                            // Test whether lhs.lower <= 0.0
                            ; tst x15, #0x1
                            ; b.eq #24 // -> end

                            // If the input interval straddles 0, then the
                            // output is [0, max(lower**2, upper**2)]
                            ; fmaxnmv s4, V(out_reg).s4
                            ; movi D(out_reg), #0
                            ; mov V(out_reg).s[1], v4.s[0]
                            ; b #8 // -> end

                            // <- swap
                            ; rev64 V(out_reg).s2, V(out_reg).s2

                            // <- end
                        ),
                        _ => unreachable!(),
                    };
                }

                ClauseOp32::AddRegReg
                | ClauseOp32::MulRegReg
                | ClauseOp32::SubRegReg
                | ClauseOp32::MinRegReg
                | ClauseOp32::MaxRegReg => {
                    let lhs_reg = ((v >> 16) & 0xFF) + REG_OFFSET;
                    let rhs_reg = ((v >> 8) & 0xFF) + REG_OFFSET;
                    let out_reg = (v & 0xFF) + REG_OFFSET;
                    match op {
                        ClauseOp32::AddRegReg => dynasm!(ops
                            ; fadd V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                        ),
                        ClauseOp32::MulRegReg => dynasm!(ops
                            // Set up v4 to contain
                            //  [lhs.upper, lhs.lower, lhs.lower, lhs.upper]
                            // and v5 to contain
                            //  [rhs.upper, rhs.lower, rhs.upper, rhs.upper]
                            //
                            // Multiplying them out will hit all four possible
                            // combinations; then we extract the min and max
                            // with vector-reducing operations
                            ; rev64 v4.s2, V(lhs_reg).s2
                            ; mov v4.d[1], V(lhs_reg).d[0]
                            ; dup v5.d2, V(rhs_reg).d[0]

                            ; fmul v4.s4, v4.s4, v5.s4
                            ; fminnmv S(out_reg), v4.s4
                            ; fmaxnmv s5, v4.s4
                            ; mov V(out_reg).s[1], v5.s[0]
                        ),
                        ClauseOp32::SubRegReg => dynasm!(ops
                            ; rev64 v4.s2, V(rhs_reg).s2
                            ; fsub V(out_reg).s2, V(lhs_reg).s2, v4.s2
                        ),
                        ClauseOp32::MinRegReg => dynasm!(ops
                            //  if lhs.upper < rhs.lower
                            //      *choices++ = CHOICE_LEFT
                            //      out = lhs
                            //  elif rhs.upper < lhs.lower
                            //      *choices++ = CHOICE_RIGHT
                            //      out = rhs
                            //  else
                            //      *choices++ = CHOICE_BOTH
                            //      out = fmin(lhs, rhs)

                            // v4 = [lhs.upper, rhs.upper]
                            // v5 = [rhs.lower, lhs.lower]
                            // This lets us do two comparisons simultaneously
                            ; zip2 v4.s2, V(lhs_reg).s2, V(rhs_reg).s2
                            ; zip1 v5.s2, V(rhs_reg).s2, V(lhs_reg).s2
                            ; fcmgt v5.s2, v5.s2, v4.s2
                            ; fmov x15, d5

                            ; tst x15, #0x1_0000_0000
                            ; b.ne #24 // -> rhs

                            ; tst x15, #0x1
                            ; b.eq #28 // -> both

                            // LHS < RHS
                            ; fmov D(out_reg), D(lhs_reg)
                            ; mov w16, #CHOICE_LEFT
                            ; b #24 // -> end

                            // <- rhs (for when RHS < LHS)
                            ; fmov D(out_reg), D(rhs_reg)
                            ; mov w16, #CHOICE_RIGHT
                            ; b #12

                            // <- both
                            ; fmin V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                            ; mov w16, #CHOICE_BOTH

                            // <- end
                            ; strb w16, [x0], #1 // post-increment
                        ),
                        ClauseOp32::MaxRegReg => dynasm!(ops
                            // Basically the same as MinRegReg
                            ; zip2 v4.s2, V(lhs_reg).s2, V(rhs_reg).s2
                            ; zip1 v5.s2, V(rhs_reg).s2, V(lhs_reg).s2
                            ; fcmgt v5.s2, v5.s2, v4.s2
                            ; fmov x15, d5

                            ; tst x15, #0x1_0000_0000
                            ; b.ne #24 // -> lhs

                            ; tst x15, #0x1
                            ; b.eq #28 // -> both

                            // LHS < RHS
                            ; fmov D(out_reg), D(rhs_reg)
                            ; mov w16, #CHOICE_RIGHT
                            ; b #24 // -> end

                            // <- lhs (when RHS < LHS)
                            ; fmov D(out_reg), D(lhs_reg)
                            ; mov w16, #CHOICE_LEFT
                            ; b #12 // -> end

                            // <- both
                            ; fmax V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                            ; mov w16, #CHOICE_BOTH

                            // <- end
                            ; strb w16, [x0], #1 // post-increment
                        ),
                        _ => unreachable!(),
                    };
                }
            }
        } else {
            let op = (v >> 16) & ((1 << 14) - 1);
            let op = ClauseOp64::from_u32(op).unwrap();
            let next = iter.next().unwrap();
            let arg_reg = ((v >> 8) & 0xFF) + REG_OFFSET;
            let out_reg = (v & 0xFF) + REG_OFFSET;
            let imm_u32 = *next;
            let imm_f32 = f32::from_bits(imm_u32);

            // Unpack the immediate using two 16-bit writes
            dynasm!(ops
                ; movz w15, #(imm_u32 >> 16), lsl 16
                ; movk w15, #(imm_u32)
                ; dup v4.s2, w15
            );
            match op {
                ClauseOp64::AddRegImm => dynasm!(ops
                    ; fadd V(out_reg).s2, V(arg_reg).s2, v4.s2
                ),
                ClauseOp64::MulRegImm => {
                    dynasm!(ops
                        ; fmul V(out_reg).s2, V(arg_reg).s2, v4.s2
                    );
                    if imm_f32 < 0.0 {
                        dynasm!(ops
                            ; rev64 V(out_reg).s2, V(out_reg).s2
                        )
                    }
                }
                ClauseOp64::SubImmReg => dynasm!(ops
                    ; rev64 v5.s2, V(arg_reg).s2
                    ; fsub V(out_reg).s2, v4.s2, v5.s2
                ),
                ClauseOp64::SubRegImm => dynasm!(ops
                    ; fsub V(out_reg).s2, V(arg_reg).s2, v4.s2
                ),
                ClauseOp64::MinRegImm => dynasm!(ops
                    //  if arg.upper < imm
                    //      *choices++ = CHOICE_LEFT
                    //      out = arg
                    //  elif imm < arg.lower
                    //      *choices++ = CHOICE_RIGHT
                    //      out = imm
                    //  else
                    //      *choices++ = CHOICE_BOTH
                    //      out = fmin(arg, imm)

                    // v4 = [imm, imm]
                    // v5 = [arg.upper, imm]
                    // v6 = [imm, arg.lower]
                    // This lets us do two comparisons simultaneously
                    ; zip2 v5.s2, V(arg_reg).s2, v4.s2
                    ; zip1 v6.s2, v4.s2, V(arg_reg).s2
                    ; fcmgt v6.s2, v6.s2, v5.s2
                    ; fmov x15, d6

                    ; tst x15, #0x1_0000_0000
                    ; b.ne >imm

                    ; tst x15, #0x1
                    ; b.eq >both

                    // arg < imm
                    ; fmov D(out_reg), D(arg_reg)
                    ; mov w16, #CHOICE_LEFT
                    ; b >end

                    // imm < arg
                    ;imm:
                    ; fmov D(out_reg), d4
                    ; mov w16, #CHOICE_RIGHT
                    ; b >end

                    ;both:
                    ; fmin V(out_reg).s2, V(arg_reg).s2, v4.s2
                    ; mov w16, #CHOICE_BOTH

                    ;end:
                    ; strb w16, [x0], #1 // post-increment
                ),
                ClauseOp64::MaxRegImm => dynasm!(ops
                    ; zip2 v5.s2, V(arg_reg).s2, v4.s2
                    ; zip1 v6.s2, v4.s2, V(arg_reg).s2
                    ; fcmgt v6.s2, v6.s2, v5.s2
                    ; fmov x15, d6

                    ; tst x15, #0x1_0000_0000
                    ; b.ne >arg

                    ; tst x15, #0x1
                    ; b.eq >both

                    // arg < imm
                    ; fmov D(out_reg), d4
                    ; mov w16, #CHOICE_RIGHT
                    ; b >end

                    // imm < arg
                    ;arg:
                    ; fmov D(out_reg), D(arg_reg)
                    ; mov w16, #CHOICE_LEFT
                    ; b >end

                    ;both:
                    ; fmax V(out_reg).s2, V(arg_reg).s2, v4.s2
                    ; mov w16, #CHOICE_BOTH

                    ;end:
                    ; strb w16, [x0], #1 // post-increment
                ),
                ClauseOp64::CopyImm => dynasm!(ops
                    ; fmov D(out_reg), x9
                ),
            };
        }
    }

    dynasm!(ops
        // Prepare our return value
        ; mov  s0, V(REG_OFFSET).s[0]
        ; mov  s1, V(REG_OFFSET).s[1]
        // Restore stack space used for spills
        ; add   sp, sp, #(stack_space)
        // Restore callee-saved floating-point registers
        ; ldp   d14, d15, [sp], #16
        ; ldp   d12, d13, [sp], #16
        ; ldp   d10, d11, [sp], #16
        ; ldp   d8, d9, [sp], #16
        // Restore frame and link register
        ; ldp   x29, x30, [sp], #16
        ; ret
    );

    let buf = ops.finalize().unwrap();
    let fn_pointer = buf.ptr(shape_fn);
    IntervalFuncHandle {
        _buf: buf,
        fn_pointer,
        choice_count: t.choice_count,
        tape: t,
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle which owns a JIT-compiled float function
pub struct FloatFuncHandle {
    _buf: dynasmrt::ExecutableBuffer,
    fn_pointer: *const u8,
}

impl FloatFuncHandle {
    pub fn get_evaluator(&self) -> FloatEval {
        FloatEval {
            fn_float: unsafe { std::mem::transmute(self.fn_pointer) },
            _p: std::marker::PhantomData,
        }
    }
}

/// Handle which owns a JIT-compiled interval function
pub struct IntervalFuncHandle<'t> {
    _buf: dynasmrt::ExecutableBuffer,
    fn_pointer: *const u8,
    choice_count: usize,
    tape: &'t Tape,
}

impl<'t> IntervalFuncHandle<'t> {
    pub fn get_evaluator(&self) -> IntervalEval {
        IntervalEval {
            fn_interval: unsafe { std::mem::transmute(self.fn_pointer) },
            choices: vec![Choice::Both; self.choice_count],
            tape: self.tape,
            _p: std::marker::PhantomData,
        }
    }
}

/// Handle for evaluation of a JIT-compiled function
///
/// The lifetime of this `struct` is bound to an `FloatFuncHandle`, which owns
/// the underlying executable memory.
pub struct FloatEval<'asm> {
    fn_float: unsafe extern "C" fn(f32, f32, f32) -> f32,
    _p: std::marker::PhantomData<&'asm ()>,
}

impl<'a> FloatEval<'a> {
    pub fn f(&self, x: f32, y: f32, z: f32) -> f32 {
        unsafe { (self.fn_float)(x, y, z) }
    }
}

/// Handle for evaluation of a JIT-compiled function
///
/// The lifetime of this `struct` is bound to an `IntervalFuncHandle`, which
/// owns the underlying executable memory.
pub struct IntervalEval<'asm> {
    fn_interval: unsafe extern "C" fn(
        [f32; 2], // X
        [f32; 2], // Y
        [f32; 2], // Z
        *mut u8,  // choices
    ) -> [f32; 2],
    choices: Vec<Choice>,
    tape: &'asm Tape,
    _p: std::marker::PhantomData<&'asm ()>,
}

impl<'a> IntervalEval<'a> {
    pub fn i(&mut self, x: [f32; 2], y: [f32; 2], z: [f32; 2]) -> [f32; 2] {
        unsafe {
            (self.fn_interval)(x, y, z, self.choices.as_mut_ptr() as *mut u8)
        }
    }

    /// Returns a simplified tape based on `self.choices`
    ///
    /// The choices array should have been calculated during the last interval
    /// evaluation.
    pub fn push(&self) -> Tape {
        self.tape.simplify(&self.choices)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::tape32::Tape,
        context::{Context, Node},
        scheduled::schedule,
    };

    fn to_float_fn(v: Node, ctx: &Context) -> FloatFuncHandle {
        let scheduled = schedule(ctx, v);
        let tape = Tape::new_with_reg_limit(&scheduled, REGISTER_LIMIT);
        build_float_fn(&tape)
    }

    fn to_tape(v: Node, ctx: &Context) -> Tape {
        let scheduled = schedule(ctx, v);
        Tape::new_with_reg_limit(&scheduled, REGISTER_LIMIT)
    }

    #[test]
    fn test_dynasm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let two = ctx.constant(2.5);
        let y2 = ctx.mul(y, two).unwrap();
        let sum = ctx.add(x, y2).unwrap();

        let jit = to_float_fn(sum, &ctx);
        let eval = jit.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 6.0);
    }

    #[test]
    fn test_interval() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let tape = to_tape(x, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_xy = |x, y| eval.i(x, y, [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [2.0, 3.0]), [0.0, 1.0]);
        assert_eq!(eval_xy([1.0, 5.0], [2.0, 3.0]), [1.0, 5.0]);

        let tape = to_tape(y, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_xy = |x, y| eval.i(x, y, [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [2.0, 3.0]), [2.0, 3.0]);
        assert_eq!(eval_xy([1.0, 5.0], [4.0, 5.0]), [4.0, 5.0]);
    }

    #[test]
    fn test_i_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let tape = to_tape(abs_x, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval([0.0, 1.0]), [0.0, 1.0]);
        assert_eq!(eval([1.0, 5.0]), [1.0, 5.0]);
        assert_eq!(eval([-2.0, 5.0]), [0.0, 5.0]);
        assert_eq!(eval([-6.0, 5.0]), [0.0, 6.0]);
        assert_eq!(eval([-6.0, -1.0]), [1.0, 6.0]);

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let tape = to_tape(sum, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_xy = |x, y| eval.i(x, y, [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 2.0]);
        assert_eq!(eval_xy([1.0, 5.0], [-2.0, 3.0]), [1.0, 8.0]);
        assert_eq!(eval_xy([1.0, 5.0], [-4.0, 3.0]), [1.0, 9.0]);
    }

    #[test]
    fn test_i_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let tape = to_tape(sqrt_x, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 4.0]), [0.0, 2.0]);
        assert_eq!(eval_x([-2.0, 4.0]), [0.0, 2.0]);
        let nanan = eval_x([-2.0, -1.0]);
        assert!(nanan[0].is_nan());
        assert!(nanan[1].is_nan());
    }

    #[test]
    fn test_i_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let tape = to_tape(sqrt_x, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 4.0]), [0.0, 16.0]);
        assert_eq!(eval_x([2.0, 4.0]), [4.0, 16.0]);
        assert_eq!(eval_x([-2.0, 4.0]), [0.0, 16.0]);
        assert_eq!(eval_x([-6.0, -2.0]), [4.0, 36.0]);
        assert_eq!(eval_x([-6.0, 1.0]), [0.0, 36.0]);
    }

    #[test]
    fn test_i_mul() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let mul = ctx.mul(x, y).unwrap();

        let tape = to_tape(mul, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_xy = |x, y| eval.i(x, y, [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [0.0, 1.0]), [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [0.0, 2.0]), [0.0, 2.0]);
        assert_eq!(eval_xy([-2.0, 1.0], [0.0, 1.0]), [-2.0, 1.0]);
        assert_eq!(eval_xy([-2.0, -1.0], [-5.0, -4.0]), [4.0, 10.0]);
        assert_eq!(eval_xy([-3.0, -1.0], [-2.0, 6.0]), [-18.0, 6.0]);
    }

    #[test]
    fn test_i_mul_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let two = ctx.constant(2.0);
        let mul = ctx.mul(x, two).unwrap();
        let tape = to_tape(mul, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [0.0, 2.0]);
        assert_eq!(eval_x([1.0, 2.0]), [2.0, 4.0]);

        let neg_three = ctx.constant(-3.0);
        let mul = ctx.mul(x, neg_three).unwrap();
        let tape = to_tape(mul, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [-3.0, 0.0]);
        assert_eq!(eval_x([1.0, 2.0]), [-6.0, -3.0]);
    }

    #[test]
    fn test_i_sub() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let sub = ctx.sub(x, y).unwrap();

        let tape = to_tape(sub, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_xy = |x, y| eval.i(x, y, [0.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [0.0, 1.0]), [-1.0, 1.0]);
        assert_eq!(eval_xy([0.0, 1.0], [0.0, 2.0]), [-2.0, 1.0]);
        assert_eq!(eval_xy([-2.0, 1.0], [0.0, 1.0]), [-3.0, 1.0]);
        assert_eq!(eval_xy([-2.0, -1.0], [-5.0, -4.0]), [2.0, 4.0]);
        assert_eq!(eval_xy([-3.0, -1.0], [-2.0, 6.0]), [-9.0, 1.0]);
    }

    #[test]
    fn test_i_sub_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let two = ctx.constant(2.0);
        let sub = ctx.sub(x, two).unwrap();
        let tape = to_tape(sub, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [-2.0, -1.0]);
        assert_eq!(eval_x([1.0, 2.0]), [-1.0, 0.0]);

        let neg_three = ctx.constant(-3.0);
        let sub = ctx.sub(neg_three, x).unwrap();
        let tape = to_tape(sub, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);
        assert_eq!(eval_x([0.0, 1.0]), [-4.0, -3.0]);
        assert_eq!(eval_x([1.0, 2.0]), [-5.0, -4.0]);
    }

    #[test]
    fn test_i_recip() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let recip = ctx.recip(x).unwrap();
        let tape = to_tape(recip, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        let mut eval_x = |x| eval.i(x, [0.0, 1.0], [0.0, 1.0]);

        let nanan = eval_x([0.0, 1.0]);
        assert!(nanan[0].is_nan());
        assert!(nanan[1].is_nan());

        let nanan = eval_x([-1.0, 0.0]);
        assert!(nanan[0].is_nan());
        assert!(nanan[1].is_nan());

        let nanan = eval_x([-2.0, 3.0]);
        assert!(nanan[0].is_nan());
        assert!(nanan[1].is_nan());

        assert_eq!(eval_x([-2.0, -1.0]), [-1.0, -0.5]);
        assert_eq!(eval_x([1.0, 2.0]), [0.5, 1.0]);
    }

    #[test]
    fn test_i_min() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let tape = to_tape(min, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        assert_eq!(eval.i([0.0, 1.0], [0.5, 1.5], [0.0, 0.0]), [0.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Both]);

        assert_eq!(eval.i([0.0, 1.0], [2.0, 3.0], [0.0, 0.0]), [0.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Left]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 1.0], [0.0, 0.0]), [0.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Right]);
    }

    #[test]
    fn test_i_min_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();

        let tape = to_tape(min, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        assert_eq!(eval.i([0.0, 1.0], [0.0, 0.0], [0.0, 0.0]), [0.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Both]);

        assert_eq!(eval.i([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]), [-1.0, 0.0]);
        assert_eq!(eval.choices, vec![Choice::Left]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 0.0], [0.0, 0.0]), [1.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Right]);
    }

    #[test]
    fn test_i_max() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let max = ctx.max(x, y).unwrap();

        let tape = to_tape(max, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        assert_eq!(eval.i([0.0, 1.0], [0.5, 1.5], [0.0, 0.0]), [0.5, 1.5]);
        assert_eq!(eval.choices, vec![Choice::Both]);

        assert_eq!(eval.i([0.0, 1.0], [2.0, 3.0], [0.0, 0.0]), [2.0, 3.0]);
        assert_eq!(eval.choices, vec![Choice::Right]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 1.0], [0.0, 0.0]), [2.0, 3.0]);
        assert_eq!(eval.choices, vec![Choice::Left]);

        let z = ctx.z();
        let max_xy_z = ctx.max(max, z).unwrap();
        let tape = to_tape(max_xy_z, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        assert_eq!(eval.i([2.0, 3.0], [0.0, 1.0], [4.0, 5.0]), [4.0, 5.0]);
        assert_eq!(eval.choices, vec![Choice::Left, Choice::Right]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 1.0], [1.0, 4.0]), [2.0, 4.0]);
        assert_eq!(eval.choices, vec![Choice::Left, Choice::Both]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 1.0], [1.0, 1.5]), [2.0, 3.0]);
        assert_eq!(eval.choices, vec![Choice::Left, Choice::Left]);
    }

    #[test]
    fn test_i_max_imm() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let one = ctx.constant(1.0);
        let max = ctx.max(x, one).unwrap();

        let tape = to_tape(max, &ctx);
        let jit = build_interval_fn(&tape);
        let mut eval = jit.get_evaluator();
        assert_eq!(eval.i([0.0, 2.0], [0.0, 0.0], [0.0, 0.0]), [1.0, 2.0]);
        assert_eq!(eval.choices, vec![Choice::Both]);

        assert_eq!(eval.i([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]), [1.0, 1.0]);
        assert_eq!(eval.choices, vec![Choice::Right]);

        assert_eq!(eval.i([2.0, 3.0], [0.0, 0.0], [0.0, 0.0]), [2.0, 3.0]);
        assert_eq!(eval.choices, vec![Choice::Left]);
    }
}
