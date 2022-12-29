use crate::{
    eval::{
        grad::{Grad, GradEvalT},
        Tape,
    },
    jit::{
        build_asm_fn, build_asm_fn_with_storage, mmap::Mmap, reg,
        AssemblerData, AssemblerT, Eval, IMM_REG, OFFSET, REGISTER_LIMIT,
    },
};
use dynasmrt::{dynasm, DynasmApi};
use std::sync::Arc;

/// Assembler for automatic differentiation / gradient evaluation
///
/// Arguments are passed as `x`, `y`, `z` in `s0-2`, and a pointer to the `vars`
/// array in `x0`.  The result is returned in `s0-3`, representing `[v, dx, dy,
/// dz]`
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`.  Each SIMD register
/// is in the order `[value, dx, dy, dz]`, e.g. the value for X is in `V0.S0`.
struct GradAssembler(AssemblerData<[f32; 4]>);

impl AssemblerT for GradAssembler {
    type Data = Grad;

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

            // Arguments are passed in S0-2; inject the derivatives here
            ; fmov s6, #1.0
            ; mov v0.S[1], v6.S[0]
            ; mov v1.S[2], v6.S[0]
            ; mov v2.S[3], v6.S[0]
        );
        out.prepare_stack(slot_count);

        Self(out)
    }
    /// Reads from `src_mem` to `dst_reg`
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(src_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; ldp D(reg(dst_reg)), d4, [x9]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        } else {
            dynasm!(self.0.ops
                ; ldp D(reg(dst_reg)), d4, [sp, #(sp_offset)]
                ; mov V(reg(dst_reg)).d[1], v4.d[0]
            )
        }
    }
    /// Writes from `src_reg` to `dst_mem`
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset = self.0.stack_pos(dst_mem);
        if sp_offset >= 512 {
            assert!(sp_offset < 4096);
            dynasm!(self.0.ops
                ; add x9, sp, #(sp_offset)
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [x9]
            )
        } else {
            dynasm!(self.0.ops
                ; mov v4.d[0], V(reg(src_reg)).d[1]
                ; stp D(reg(src_reg)), d4, [sp, #(sp_offset)]
            )
        }
    }
    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops ; mov V(reg(out_reg)).b16, V(src_arg as u32).b16);
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        assert!(src_arg * 4 < 16384);
        dynasm!(self.0.ops
            ; ldr S(reg(out_reg)), [x0, #(src_arg * 4)]
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

    fn build_fma(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // We can't really take advantage of the FMA here, so we'll copy
        // the multiplication code from above (storing the result in v6.s4)
        // then accumulate with a plain `fadd`
        dynasm!(self.0.ops
            ; dup v6.s4, V(reg(lhs_reg)).s[0]
            ; fmul v5.s4, v6.s4, V(reg(rhs_reg)).s4
            ; fmov s7, s5
            ; dup v6.s4, V(reg(rhs_reg)).s[0]
            ; fmla v5.s4, v6.s4, V(reg(lhs_reg)).s4

            ; mov v6.b16, v5.b16
            ; mov v6.s[0], v7.s[0]

            ; fadd V(reg(out_reg)).s4, V(reg(out_reg)).s4, v6.s4
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
            ; b.gt #12 // -> lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b #8 // -> end
            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            // end:
        )
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fcmp S(reg(lhs_reg)), S(reg(rhs_reg))
            ; b.lt #12 // -> lhs
            // Happy path: v >= 0, so we just copy the register
            ; mov V(reg(out_reg)).b16, V(reg(rhs_reg)).b16
            ; b #8 // -> end
            // lhs:
            ; mov V(reg(out_reg)).b16, V(reg(lhs_reg)).b16
            // end:
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

    fn finalize(mut self, out_reg: u8) -> Mmap {
        dynasm!(self.0.ops
            // Prepare our return value, writing into s0-3
            ; mov s0, V(reg(out_reg)).s[0]
            ; mov s1, V(reg(out_reg)).s[1]
            ; mov s2, V(reg(out_reg)).s[2]
            ; mov s3, V(reg(out_reg)).s[3]

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

/// Evaluator for a JIT-compiled function performing gradient evaluation.
pub struct JitGradEval {
    mmap: Arc<Mmap>,
    var_count: usize,
    /// X, Y, Z are passed by value; the output is written to an array of 4
    /// floats (allocated by the caller)
    fn_grad: unsafe extern "C" fn(f32, f32, f32, *const f32) -> [f32; 4],
}

impl GradEvalT<Eval> for JitGradEval {
    type Storage = Mmap;

    fn new(t: &Tape<Eval>) -> Self {
        assert_eq!(t.reg_limit(), REGISTER_LIMIT);
        let mmap = build_asm_fn::<GradAssembler>(t);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_grad: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn new_with_storage(t: &Tape<Eval>, prev: Self::Storage) -> Self {
        let mmap = build_asm_fn_with_storage::<GradAssembler>(t, prev);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_grad: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.mmap).ok()
    }
    fn eval_f(&mut self, x: f32, y: f32, z: f32, vars: &[f32]) -> Grad {
        assert_eq!(vars.len(), self.var_count);
        let [v, x, y, z] = unsafe { (self.fn_grad)(x, y, z, vars.as_ptr()) };
        Grad::new(v, x, y, z)
    }
}
