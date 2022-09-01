use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};
use num_traits::FromPrimitive;

use crate::backend::tape32::{ClauseOp32, ClauseOp64, Tape};

/// We can use registers v8-v15 (callee saved) and v16-v31 (caller saved)
pub const REGISTER_LIMIT: usize = 24;

pub fn from_tape(t: &Tape) -> AsmHandle {
    assert_eq!(t.fast_reg_limit, REGISTER_LIMIT);

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    dynasm!(ops
    ; -> shape_fn:
    );
    let shape_fn = ops.offset();

    let stack_space =
        t.num_registers.saturating_sub(t.fast_reg_limit) as u32 * 4;
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

    const FAST_REG_OFFSET: u32 = 8;

    let mut iter = t.tape.iter();
    while let Some(v) = iter.next() {
        if v & (1 << 30) == 0 {
            let op = (v >> 24) & ((1 << 6) - 1);
            let op = ClauseOp32::from_u32(op).unwrap();
            match op {
                ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                    let fast_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    let extended_reg = (v >> 8) & 0xFFFF;
                    let sp_offset =
                        4 * (extended_reg - t.fast_reg_limit as u32);
                    match op {
                        ClauseOp32::Load => {
                            dynasm!(ops
                                ; ldr S(fast_reg), [sp, #(sp_offset)]
                            );
                        }
                        ClauseOp32::Store => {
                            dynasm!(ops
                                ; str S(fast_reg), [sp, #(sp_offset)]
                            );
                        }
                        ClauseOp32::Swap => {
                            dynasm!(ops
                                ; fmov s3, S(fast_reg)
                                ; ldr S(fast_reg), [sp, #(sp_offset)]
                                ; str s3, [sp, #(sp_offset)]
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                ClauseOp32::Input => {
                    let input = (v >> 16) & 0xFF;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    dynasm!(ops
                        ; fmov S(out_reg), S(input)
                    );
                }
                ClauseOp32::CopyReg
                | ClauseOp32::NegReg
                | ClauseOp32::AbsReg
                | ClauseOp32::RecipReg
                | ClauseOp32::SqrtReg
                | ClauseOp32::SquareReg => {
                    let lhs_reg = ((v >> 16) & 0xFF) + FAST_REG_OFFSET;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    match op {
                        ClauseOp32::CopyReg => dynasm!(ops
                            ; fmov S(out_reg), S(lhs_reg)
                        ),
                        ClauseOp32::NegReg => dynasm!(ops
                            ; fneg S(out_reg), S(lhs_reg)
                        ),
                        ClauseOp32::AbsReg => dynasm!(ops
                            ; fabs S(out_reg), S(lhs_reg)
                        ),
                        // TODO: is reciprocal estimate okay, or should we do
                        // division?
                        ClauseOp32::RecipReg => dynasm!(ops
                            ; frecpe S(out_reg), S(lhs_reg)
                        ),
                        ClauseOp32::SqrtReg => dynasm!(ops
                            ; fsqrt S(out_reg), S(lhs_reg)
                        ),
                        ClauseOp32::SquareReg => dynasm!(ops
                            ; fmul S(out_reg), S(lhs_reg), S(lhs_reg)
                        ),
                        _ => unreachable!(),
                    };
                }

                ClauseOp32::AddRegReg
                | ClauseOp32::MulRegReg
                | ClauseOp32::SubRegReg
                | ClauseOp32::MinRegReg
                | ClauseOp32::MaxRegReg => {
                    let lhs_reg = ((v >> 16) & 0xFF) + FAST_REG_OFFSET;
                    let rhs_reg = ((v >> 8) & 0xFF) + FAST_REG_OFFSET;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    match op {
                        ClauseOp32::AddRegReg => dynasm!(ops
                            ; fadd S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::MulRegReg => dynasm!(ops
                            ; fmul S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::SubRegReg => dynasm!(ops
                            ; fsub S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::MinRegReg => dynasm!(ops
                            ; fmin S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::MaxRegReg => dynasm!(ops
                            ; fmax S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::Load | ClauseOp32::Store => {
                            unreachable!()
                        }
                        _ => unreachable!(),
                    };
                }
                _ => panic!("Bad 32-bit opcode"),
            }
        } else {
            let op = (v >> 16) & ((1 << 14) - 1);
            let op = ClauseOp64::from_u32(op).unwrap();
            let next = iter.next().unwrap();
            let arg_reg = ((v >> 8) & 0xFF) + FAST_REG_OFFSET;
            let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
            let imm_u32 = *next;

            // Unpack the immediate using two 16-bit writes
            dynasm!(ops
                ; movz w9, #(imm_u32 >> 16), lsl 16
                ; movk w9, #(imm_u32)
                ; fmov s3, w9
            );
            match op {
                ClauseOp64::AddRegImm => dynasm!(ops
                    ; fadd S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::MulRegImm => dynasm!(ops
                    ; fmul S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::SubImmReg => dynasm!(ops
                    ; fsub S(out_reg), s3, S(arg_reg)
                ),
                ClauseOp64::SubRegImm => dynasm!(ops
                    ; fsub S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::MinRegImm => dynasm!(ops
                    ; fmin S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::MaxRegImm => dynasm!(ops
                    ; fmax S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::CopyImm => dynasm!(ops
                    ; fmov S(out_reg), w9
                ),
            };
        }
    }

    dynasm!(ops
        // Prepare our return value
        ; fmov  s0, S(FAST_REG_OFFSET)
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
    let shape_fn_pointer = buf.ptr(shape_fn);
    AsmHandle {
        _buf: buf,
        shape_fn_pointer,
    }
}

pub fn tape_to_interval(t: &Tape) -> AsmHandle {
    assert_eq!(t.fast_reg_limit, REGISTER_LIMIT);

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    dynasm!(ops
    ; -> shape_fn:
    );
    let shape_fn = ops.offset();

    let stack_space =
        t.num_registers.saturating_sub(t.fast_reg_limit) as u32 * 4 * 2;
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

        // Arguments are passed in S0-3; collect them into V0-1
        ; ins v0.s[1], v1.s[0]
        ; ins v1.s[0], v2.s[0]
        ; ins v1.s[1], v3.s[0]
    );

    const FAST_REG_OFFSET: u32 = 8;

    let mut iter = t.tape.iter();
    while let Some(v) = iter.next() {
        if v & (1 << 30) == 0 {
            let op = (v >> 24) & ((1 << 6) - 1);
            let op = ClauseOp32::from_u32(op).unwrap();
            match op {
                ClauseOp32::Load | ClauseOp32::Store | ClauseOp32::Swap => {
                    let fast_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    let extended_reg = (v >> 8) & 0xFFFF;
                    let sp_offset =
                        2 * 4 * (extended_reg - t.fast_reg_limit as u32);
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
                                ; fmov d3, D(fast_reg)
                                ; ldr D(fast_reg), [sp, #(sp_offset)]
                                ; str s3, [sp, #(sp_offset)]
                            );
                        }
                        _ => unreachable!(),
                    }
                }
                ClauseOp32::Input => {
                    let input = (v >> 16) & 0xFF;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
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
                    let lhs_reg = ((v >> 16) & 0xFF) + FAST_REG_OFFSET;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    match op {
                        ClauseOp32::CopyReg => dynasm!(ops
                            ; fmov D(out_reg), D(lhs_reg)
                        ),
                        ClauseOp32::NegReg => dynasm!(ops
                            ; fneg V(out_reg).s2, V(lhs_reg).s2
                            ; rev64 V(out_reg).s2, V(out_reg).s2
                        ),
                        ClauseOp32::AbsReg => dynasm!(ops
                            // Store lhs < 0.0 in x8
                            ; fcmle v3.s2, V(lhs_reg).s2, #0.0
                            ; fmov x8, d3

                            // Store abs(lhs) in V(out_reg)
                            ; fabs V(out_reg).s2, V(lhs_reg).s2

                            // Check whether lhs.upper < 0
                            ; tst x8, #0x1_0000_0000
                            ; b.ne >upper_lz // ne is !Z

                            // Check whether lhs.lower < 0
                            ; tst x8, #0x1
                            ; b.ne >lower_lz

                            // otherwise, we're good; return the original
                            ; b >end

                            // if upper < 0
                            //   return [-upper, -lower]
                            ;upper_lz:
                            ; rev64 V(out_reg).s2, V(out_reg).s2
                            ; b >end

                            // if lower < 0
                            //   return [0, max(abs(upper), abs(lower))]
                            ;lower_lz:
                            ; fmaxnmv s7, V(out_reg).s4
                            ; movi D(out_reg), #0
                            ; ins V(out_reg).s[1], v7.s[0]

                            ;end:
                        ),
                        ClauseOp32::RecipReg => dynasm!(ops
                            // TODO
                            ; frecpe S(out_reg), S(lhs_reg)
                        ),
                        ClauseOp32::SqrtReg => dynasm!(ops
                            // Store lhs < 0.0 in x8
                            ; fcmle v3.s2, V(lhs_reg).s2, #0.0
                            ; fmov x8, d3

                            // Check whether lhs.upper < 0
                            ; tst x8, #0x1_0000_0000
                            ; b.ne >upper_lz // ne is Z == 0

                            ; tst x8, #0x1
                            ; b.ne >lower_lz // ne is Z == 0

                            // Happy path
                            ; fsqrt V(out_reg).s2, V(lhs_reg).s2
                            ; b >end

                            ;lower_lz:
                            ; ins v4.s[0], V(lhs_reg).s[1]
                            ; fsqrt s4, s4
                            ; movi D(out_reg), #0
                            ; ins V(out_reg).s[1], v4.s[0]
                            ; b >end

                            ;upper_lz:
                            // Make V(out_reg) NaN by doing an illegal division
                            // (XXX optimize this)
                            ; movi D(out_reg), #0
                            ; fdiv V(out_reg).s2, V(out_reg).s2, V(out_reg).s2
                            // Fallthrough

                            ;end:
                        ),
                        ClauseOp32::SquareReg => dynasm!(ops
                            // Store lhs < 0.0 in x8
                            ; fcmle v3.s2, V(lhs_reg).s2, #0.0
                            ; fmov x8, d3
                            ; fmul V(out_reg).s2, V(lhs_reg).s2, V(lhs_reg).s2

                            // Check whether lhs.upper < 0
                            ; tst x8, #0x1_0000_0000
                            ; b.ne >swap // ne is Z == 0

                            ; tst x8, #0x1
                            ; b.ne >split // ne is Z == 0

                            // Happy path
                            ; b >end

                            ;swap:
                            ; rev64 V(out_reg).s2, V(out_reg).s2
                            ; b >end

                            ;split:
                            // If the input interval straddles 0, then the
                            // output is [0, max(lower**2, upper**2)]
                            ; fmaxnmv s4, V(out_reg).s4
                            ; movi D(out_reg), #0
                            ; ins V(out_reg).s[1], v4.s[0]
                            ; b >end

                            ;end:
                        ),
                        _ => unreachable!(),
                    };
                }

                ClauseOp32::AddRegReg
                | ClauseOp32::MulRegReg
                | ClauseOp32::SubRegReg
                | ClauseOp32::MinRegReg
                | ClauseOp32::MaxRegReg => {
                    let lhs_reg = ((v >> 16) & 0xFF) + FAST_REG_OFFSET;
                    let rhs_reg = ((v >> 8) & 0xFF) + FAST_REG_OFFSET;
                    let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
                    match op {
                        ClauseOp32::AddRegReg => dynasm!(ops
                            ; fadd V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                        ),
                        ClauseOp32::MulRegReg => dynasm!(ops
                            // TODO
                            ; fmul S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::SubRegReg => dynasm!(ops
                            // TODO
                            ; fsub S(out_reg), S(lhs_reg), S(rhs_reg)
                        ),
                        ClauseOp32::MinRegReg => dynasm!(ops
                            ; fmin V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                        ),
                        ClauseOp32::MaxRegReg => dynasm!(ops
                            ; fmax V(out_reg).s2, V(lhs_reg).s2, V(rhs_reg).s2
                        ),
                        ClauseOp32::Load | ClauseOp32::Store => {
                            unreachable!()
                        }
                        _ => unreachable!(),
                    };
                }
                _ => panic!("Bad 32-bit opcode"),
            }
        } else {
            let op = (v >> 16) & ((1 << 14) - 1);
            let op = ClauseOp64::from_u32(op).unwrap();
            let next = iter.next().unwrap();
            let arg_reg = ((v >> 8) & 0xFF) + FAST_REG_OFFSET;
            let out_reg = (v & 0xFF) + FAST_REG_OFFSET;
            let imm_u32 = *next;

            // Unpack the immediate using two 16-bit writes
            dynasm!(ops
                ; movz w9, #(imm_u32 >> 16), lsl 16
                ; movk w9, #(imm_u32)
                ; dup v3.s2, w9
            );
            match op {
                ClauseOp64::AddRegImm => dynasm!(ops
                    ; fadd V(out_reg).s2, V(arg_reg).s2, v3.s2
                ),
                ClauseOp64::MulRegImm => dynasm!(ops
                    // TODO
                    ; fmul S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::SubImmReg => dynasm!(ops
                    // TODO
                    ; fsub S(out_reg), S(arg_reg), s3
                ),
                ClauseOp64::SubRegImm => dynasm!(ops
                    ; fsub V(out_reg).s2, V(arg_reg).s2, v3.s2
                ),
                ClauseOp64::MinRegImm => dynasm!(ops
                    ; fmin V(out_reg).s2, V(arg_reg).s2, v3.s2
                ),
                ClauseOp64::MaxRegImm => dynasm!(ops
                    ; fmax V(out_reg).s2, V(arg_reg).s2, v3.s2
                ),
                ClauseOp64::CopyImm => dynasm!(ops
                    ; fmov D(out_reg), x9
                ),
            };
        }
    }

    dynasm!(ops
        // Prepare our return value
        ; mov  s0, V(FAST_REG_OFFSET).s[0]
        ; mov  s1, V(FAST_REG_OFFSET).s[1]
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
    let shape_fn_pointer = buf.ptr(shape_fn);
    AsmHandle {
        _buf: buf,
        shape_fn_pointer,
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle which owns JITed functions
pub struct AsmHandle {
    _buf: dynasmrt::ExecutableBuffer,
    shape_fn_pointer: *const u8,
}

impl AsmHandle {
    pub fn to_eval(&self) -> AsmEval<'_> {
        let f = unsafe { std::mem::transmute(self.shape_fn_pointer) };
        AsmEval {
            fn_float: f,
            _p: std::marker::PhantomData,
        }
    }
    pub fn to_interval(&self) -> IntervalEval<'_> {
        let f = unsafe { std::mem::transmute(self.shape_fn_pointer) };
        IntervalEval {
            fn_interval: f,
            _p: std::marker::PhantomData,
        }
    }
}

type FloatFunc = unsafe extern "C" fn(
    f32, // X
    f32, // Y
) -> f32;

type IntervalFunc = unsafe extern "C" fn(
    [f32; 2], // X
    [f32; 2], // Y
) -> [f32; 2];

/// Handle for evaluation of JITed functions
///
/// The lifetime of this `struct` is bound to an `AsmHandle`, which owns the
/// underlying executable memory.
pub struct AsmEval<'asm> {
    fn_float: FloatFunc,
    _p: std::marker::PhantomData<&'asm ()>,
}

impl<'a> AsmEval<'a> {
    pub fn eval(&self, x: f32, y: f32) -> f32 {
        unsafe { (self.fn_float)(x, y) }
    }
}

pub struct IntervalEval<'asm> {
    fn_interval: IntervalFunc,
    _p: std::marker::PhantomData<&'asm ()>,
}

impl<'a> IntervalEval<'a> {
    pub fn eval(&self, x: [f32; 2], y: [f32; 2]) -> [f32; 2] {
        unsafe { (self.fn_interval)(x, y) }
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

    fn to_float_fn(v: Node, ctx: &Context) -> AsmHandle {
        let scheduled = schedule(ctx, v);
        let tape = Tape::new_with_reg_limit(&scheduled, REGISTER_LIMIT);
        from_tape(&tape)
    }

    fn to_interval_fn(v: Node, ctx: &Context) -> AsmHandle {
        let scheduled = schedule(ctx, v);
        let tape = Tape::new_with_reg_limit(&scheduled, REGISTER_LIMIT);
        tape_to_interval(&tape)
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
        let eval = jit.to_eval();
        assert_eq!(eval.eval(1.0, 2.0), 6.0);
    }

    #[test]
    fn test_interval() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();

        let jit = to_interval_fn(x, &ctx);
        let eval = jit.to_interval();
        assert_eq!(eval.eval([0.0, 1.0], [2.0, 3.0]), [0.0, 1.0]);
        assert_eq!(eval.eval([1.0, 5.0], [2.0, 3.0]), [1.0, 5.0]);

        let jit = to_interval_fn(y, &ctx);
        let eval = jit.to_interval();
        assert_eq!(eval.eval([0.0, 1.0], [2.0, 3.0]), [2.0, 3.0]);
        assert_eq!(eval.eval([1.0, 5.0], [4.0, 5.0]), [4.0, 5.0]);
    }

    #[test]
    fn test_i_abs() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let abs_x = ctx.abs(x).unwrap();

        let jit = to_interval_fn(abs_x, &ctx);
        let eval = jit.to_interval();
        let y = [0.0, 1.0];
        assert_eq!(eval.eval([0.0, 1.0], y), [0.0, 1.0]);
        assert_eq!(eval.eval([1.0, 5.0], y), [1.0, 5.0]);
        assert_eq!(eval.eval([-2.0, 5.0], y), [0.0, 5.0]);
        assert_eq!(eval.eval([-6.0, 5.0], y), [0.0, 6.0]);
        assert_eq!(eval.eval([-6.0, -1.0], y), [1.0, 6.0]);

        let y = ctx.y();
        let abs_y = ctx.abs(y).unwrap();
        let sum = ctx.add(abs_x, abs_y).unwrap();
        let jit = to_interval_fn(sum, &ctx);
        let eval = jit.to_interval();
        assert_eq!(eval.eval([0.0, 1.0], [0.0, 1.0]), [0.0, 2.0]);
        assert_eq!(eval.eval([1.0, 5.0], [-2.0, 3.0]), [1.0, 8.0]);
        assert_eq!(eval.eval([1.0, 5.0], [-4.0, 3.0]), [1.0, 9.0]);
    }

    #[test]
    fn test_i_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.sqrt(x).unwrap();

        let jit = to_interval_fn(sqrt_x, &ctx);
        let eval = jit.to_interval();
        let y = [0.0, 1.0];
        assert_eq!(eval.eval([0.0, 1.0], y), [0.0, 1.0]);
        assert_eq!(eval.eval([0.0, 4.0], y), [0.0, 2.0]);
        assert_eq!(eval.eval([-2.0, 4.0], y), [0.0, 2.0]);
        let nanan = eval.eval([-2.0, -1.0], y);
        assert!(nanan[0].is_nan());
        assert!(nanan[1].is_nan());
    }

    #[test]
    fn test_i_square() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let sqrt_x = ctx.square(x).unwrap();

        let jit = to_interval_fn(sqrt_x, &ctx);
        let eval = jit.to_interval();
        let y = [0.0, 1.0];
        assert_eq!(eval.eval([0.0, 1.0], y), [0.0, 1.0]);
        assert_eq!(eval.eval([0.0, 4.0], y), [0.0, 16.0]);
        assert_eq!(eval.eval([2.0, 4.0], y), [4.0, 16.0]);
        assert_eq!(eval.eval([-2.0, 4.0], y), [0.0, 16.0]);
        assert_eq!(eval.eval([-6.0, -2.0], y), [4.0, 36.0]);
        assert_eq!(eval.eval([-6.0, 1.0], y), [0.0, 36.0]);
    }
}
