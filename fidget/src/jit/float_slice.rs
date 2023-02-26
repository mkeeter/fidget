use crate::jit::{
    mmap::Mmap, reg, AssemblerData, AssemblerT, Error, JitBulkEval,
    SimdAssembler, IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi};

#[cfg(target_arch = "aarch64")]
const SIMD_WIDTH: usize = 4;

#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "x86_64")]
use dynasmrt::DynasmLabelApi;

pub struct FloatSliceAssembler(AssemblerData<[f32; SIMD_WIDTH]>, usize);

/// Assembler for SIMD point-wise evaluation.
///
/// Arguments are passed as 3x `*const f32` in `x0-2`, a var array in
/// `x3`, and an output array `*mut f32` in `x4`.  Each pointer in the input
/// and output arrays represents 4x `f32`; the var array is single `f32`s
///
/// During evaluation, X, Y, and Z are stored in `V0-3.S4`
#[cfg(target_arch = "aarch64")]
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
        let loop_start = out.ops.len();

        dynasm!(out.ops
            // The loop returns here, and we check whether we need to loop
            // Remember, at this point we have
            //  x0: x input array pointer
            //  x1: y input array pointer
            //  x2: z input array pointer
            //  x3: vars input array pointer (non-advancing)
            //  x4: output array pointer
            //  x5: number of points to evaluate
            //
            // We'll be advancing x0, x1, x2 here (and decrementing x5 by 4);
            // x4 is advanced in finalize().

            ; cmp x5, #0
            ; b.ne #32 // skip to loop body

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
        );

        Self(out, loop_start)
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

    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Prepare our return value, writing to the pointer in x4
            // It's fine to overwrite X at this point in V0, since we're not
            // using it anymore.
            ; mov v0.d[0], V(reg(out_reg)).d[1]
            ; stp D(reg(out_reg)), d0, [x4], #16
        );
        let jump_size: i32 = (self.0.ops.len() - self.1).try_into().unwrap();
        assert!(jump_size.abs() < (1 << 25));
        dynasm!(self.0.ops
            ; b #-jump_size
        );

        self.0.ops.finalize()
    }
}

/// Assembler for SIMD point-wise evaluation.
///
/// Arguments are passed as follows:
///
/// | Argument | Register | Type         |
/// | ---------|----------|--------------|
/// | X        | `rdi`    | `*const f32` |
/// | Y        | `rsi`    | `*const f32` |
/// | Z        | `rdx`    | `*const f32` |
/// | vars     | `rcx`    | `*mut f32`   |
/// | out      | `r8`     | `*mut f32`   |
/// | size     | `r9`     | `u64`        |
///
/// The arrays must be an even multiple of 8 floats, since we're using AVX2 and
/// 256-bit wide operations for everything.
///
/// During evaluation, X, Y, and Z values are stored on the stack to keep
/// registers unoccupied.
#[cfg(target_arch = "x86_64")]
impl AssemblerT for FloatSliceAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
        );
        out.prepare_stack(slot_count);
        dynasm!(out.ops
            // The loop returns here, and we check whether to keep looping
            ; ->loop_start:

            ; test r9, r9
            ; jnz >body

            // Finalization code, which happens after all evaluation is complete
            ; add rsp, out.mem_offset as i32
            ; pop rbp
            ; ret

            ; body:
            // Copy from the input pointers into the stack right below rbp
            ; vmovups ymm0, [rdi]
            ; vmovups [rbp - 32], ymm0
            ; add rdi, 32

            ; vmovups ymm0, [rsi]
            ; vmovups [rbp - 64], ymm0
            ; add rsi, 32

            ; vmovups ymm0, [rdx]
            ; vmovups [rbp - 96], ymm0
            ; add rdx, 32
        );
        // We use a global label instead of a specific address, since x86
        // encoding computing the exact jump awkward; let the library do it for
        // us instead.
        Self(out, 0)
    }
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!(dst_reg < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(src_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovups Ry(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!(src_reg < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(dst_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovups [rsp + sp_offset], Ry(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops
            ; vmovups Ry(reg(out_reg)), [rbp - 32 * (src_arg as i32 + 1)]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; movss Rx(reg(out_reg)), [rcx + 4 * (src_arg as i32)]
            ; vbroadcastss Ry(reg(out_reg)), Rx(reg(out_reg))
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovups Ry(reg(out_reg)), Ry(reg(lhs_reg))
        );
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; mov eax, 0x80000000u32 as i32
            ; movd Rx(IMM_REG), eax
            ; vbroadcastss Ry(IMM_REG), Rx(IMM_REG)
            ; vpxor Ry(reg(out_reg)), Ry(IMM_REG), Ry(reg(lhs_reg))
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; mov eax, 0x7fffffffu32 as i32
            ; movd Rx(IMM_REG), eax
            ; vbroadcastss Ry(IMM_REG), Rx(IMM_REG)
            ; vpand Ry(reg(out_reg)), Ry(IMM_REG), Ry(reg(lhs_reg))
        );
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        let imm = self.load_imm(1.0);
        dynasm!(self.0.ops
            ; vdivps Ry(reg(out_reg)), Ry(reg(imm)), Ry(reg(lhs_reg))
        );
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vsqrtps Ry(reg(out_reg)), Ry(reg(lhs_reg))
        );
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(lhs_reg))
        );
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vaddps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vsubps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vdivps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // TODO: does this handle NaN correctly?
        dynasm!(self.0.ops
            ; vmaxps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // TODO: does this handle NaN correctly?
        dynasm!(self.0.ops
            ; vminps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))
        );
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; mov eax, imm_u32 as i32
            ; movd Rx(IMM_REG), eax
            ; vbroadcastss Ry(IMM_REG), Rx(IMM_REG)
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Copy data from out_reg into the out array, then adjust it
            ; vmovups [r8], Ry(reg(out_reg))
            ; add r8, 32
            ; sub r9, 8
            ; jmp ->loop_start
        );

        self.0.ops.finalize()
    }
}

impl SimdAssembler for FloatSliceAssembler {
    const SIMD_SIZE: usize = SIMD_WIDTH;
}

////////////////////////////////////////////////////////////////////////////////

pub type JitFloatSliceEval = JitBulkEval<FloatSliceAssembler>;
