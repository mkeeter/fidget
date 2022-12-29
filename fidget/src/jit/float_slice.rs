use crate::{
    eval::{float_slice::FloatSliceEvalT, Tape},
    jit::{
        build_asm_fn, build_asm_fn_with_storage, mmap::Mmap, reg,
        AssemblerData, AssemblerT, Eval, IMM_REG, OFFSET, REGISTER_LIMIT,
    },
};
use dynasmrt::{dynasm, DynasmApi};
use std::sync::Arc;

/// Assembler for SIMD point-wise evaluation.
///
/// Arguments are passed as 3x `*const f32` in `x0-2`, a var array in
/// `x3`, and an output array `*mut f32` in `x4`.  Each pointer in the input
/// and output arrays represents 4x `f32`; the var array is single `f32`s
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`
struct FloatSliceAssembler(AssemblerData<[f32; 4]>);

impl AssemblerT for FloatSliceAssembler {
    type Data = f32;

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

        );
        out.prepare_stack(slot_count);

        dynasm!(out.ops
            ; b #8 // Skip the call in favor of setup

            // call:
            ; bl #72 // -> func

            // The function returns here, and we check whether we need to loop
            // Remember, at this point we have
            //  x0: x input array pointer
            //  x1: y input array pointer
            //  x2: z input array pointer
            //  x3: vars input array pointer (non-advancing)
            //  x4: output array pointer
            //  x5: number of points to evaluate
            //
            // We'll be advancing x0, x1, x2 here (and decrementing x5 by 4);
            // x3 is advanced in finalize().

            ; cmp x5, #0
            ; b.eq #36 // -> fini
            ; sub x5, x5, #4 // We handle 4 items at a time

            // Load V0/1/2.S4 with X/Y/Z values, post-increment
            //
            // We're actually loading two f32s, but we can pretend they're
            // doubles in order to move 64 bits at a time
            ; ldp d0, d1, [x0], #16
            ; mov v0.d[1], v1.d[0]
            ; ldp d1, d2, [x1], #16
            ; mov v1.d[1], v2.d[0]
            ; ldp d2, d3, [x2], #16
            ; mov v2.d[1], v3.d[0]

            ; b #-40 // -> call

            // fini:
            // This is our finalization code, which happens after all evaluation
            // is complete.
            //
            // Restore stack space used for spills
            ; add   sp, sp, #(out.mem_offset as u32)
            // Restore callee-saved floating-point registers
            ; ldp   d14, d15, [sp], #16
            ; ldp   d12, d13, [sp], #16
            ; ldp   d10, d11, [sp], #16
            ; ldp   d8, d9, [sp], #16
            // Restore frame and link register
            ; ldp   x29, x30, [sp], #16
            ; ret

            // func:
        );

        Self(out)
    }
    /// Reads from `src_mem` to `dst_reg`, using D4 as an intermediary
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

    /// Writes from `src_reg` to `dst_mem`, using D4 as an intermediary
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
            ; ldr w15, [x3, #(src_arg * 4)]
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
    fn build_fma(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; fmla V(reg(out_reg)).s4, V(reg(lhs_reg)).s4, V(reg(rhs_reg)).s4
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

    fn finalize(mut self, out_reg: u8) -> Mmap {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x3
            // It's fine to overwrite X at this point in V0, since we're not
            // using it anymore.
            ; mov v0.d[0], V(reg(out_reg)).d[1]
            ; stp D(reg(out_reg)), d0, [x4], #16
            ; ret
        );

        self.0.ops.finalize()
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for a JIT-compiled function taking `[f32; 4]` SIMD values
pub struct JitFloatSliceEval {
    mmap: Arc<Mmap>,
    var_count: usize,
    fn_vec: unsafe extern "C" fn(
        *const f32, // X
        *const f32, // Y
        *const f32, // Z
        *const f32, // vars
        *mut f32,   // out
        u64,        // size
    ),
}

impl FloatSliceEvalT<Eval> for JitFloatSliceEval {
    type Storage = Mmap;

    fn new(t: &Tape<Eval>) -> Self {
        let mmap = build_asm_fn::<FloatSliceAssembler>(t);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_vec: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn new_with_storage(t: &Tape<Eval>, prev: Self::Storage) -> Self {
        let mmap = build_asm_fn_with_storage::<FloatSliceAssembler>(t, prev);
        let ptr = mmap.as_ptr();
        JitFloatSliceEval {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_vec: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.mmap).ok()
    }

    fn eval_s(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [f32],
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.var_count);

        let n = xs.len();

        // Special case for < 4 items, in which case the input slices can't be
        // used as workspace (because we need at least 4x f32)
        if n < 4 {
            let mut x = [0.0; 4];
            let mut y = [0.0; 4];
            let mut z = [0.0; 4];
            x[0..n].copy_from_slice(xs);
            y[0..n].copy_from_slice(ys);
            z[0..n].copy_from_slice(zs);
            let mut tmp = [std::f32::NAN; 4];
            unsafe {
                (self.fn_vec)(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vars.as_ptr(),
                    tmp.as_mut_ptr(),
                    4,
                );
            }
            out[0..n].copy_from_slice(&tmp[0..n]);
        } else {
            // Our vectorized function only accepts set of 4 values, so we'll
            // find the biggest multiple of four, then do an extra operation
            // to process any remainders.

            let m = (n / 4) * 4; // Round down
            unsafe {
                (self.fn_vec)(
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    vars.as_ptr(),
                    out.as_mut_ptr(),
                    m as u64,
                );
            }
            // If we weren't given a multiple of 4, then we'll handle the
            // remaining 1-3 items by simply evaluating the *last* 4 items
            // in the array again.
            if n != m {
                unsafe {
                    (self.fn_vec)(
                        xs.as_ptr().add(n - 4),
                        ys.as_ptr().add(n - 4),
                        zs.as_ptr().add(n - 4),
                        vars.as_ptr(),
                        out.as_mut_ptr().add(n - 4),
                        4,
                    );
                }
            }
        }
    }
}
