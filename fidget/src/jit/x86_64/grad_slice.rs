use crate::{
    eval::types::Grad,
    jit::{
        grad_slice::GradSliceAssembler, mmap::Mmap, reg, Assembler,
        AssemblerData, IMM_REG, OFFSET, REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation for the gradient slice assembler on `x86_64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register | Type               |
/// |------------|----------|--------------------|
/// | X          | `rdi`    | `*const f32`       |
/// | Y          | `rsi`    | `*const f32`       |
/// | Z          | `rdx`    | `*const f32`       |
/// | `vars`     | `rcx`    | `*const f32`       |
/// | `out`      | `r8`     | `*const [f32; 4]`  |
/// | `count`    | `r9`     | `u64`              |
///
/// During evaluation, X, Y, and Z values are stored on the stack to keep
/// registers unoccupied.
impl Assembler for GradSliceAssembler {
    type Data = Grad;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
            ; vzeroupper
        );
        out.prepare_stack(slot_count);
        dynasm!(out.ops
            // The loop returns here, and we check whether to keep looping
            ; ->L:

            ; test r9, r9
            ; jz ->X // jump to the exit if we're done, otherwise fallthrough

            // Copy from the input pointers into the stack right below rbp
            ; mov eax, 1.0f32.to_bits() as i32
            ; mov [rbp - 12], eax  // 1
            ; mov [rbp - 24], eax // 1
            ; mov [rbp - 36], eax // 1

            ; mov eax, [rdi]
            ; mov [rbp - 16], eax  // X
            ; add rdi, 4

            ; mov eax, [rsi]
            ; mov [rbp - 32], eax // Y
            ; add rsi, 4

            ; mov eax, [rdx]
            ; mov [rbp - 48], eax // Z
            ; add rdx, 4

            ; mov eax, 0.0f32.to_bits() as i32
            ; mov [rbp - 8], eax // 0
            ; mov [rbp - 4], eax // 0
            ; mov [rbp - 28], eax // 0
            ; mov [rbp - 20], eax // 0
            ; mov [rbp - 40], eax // 0
            ; mov [rbp - 44], eax // 0
        );
        Self(out)
    }
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(src_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovups Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(dst_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovups [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops
            ; vmovups Rx(reg(out_reg)), [rbp - 16 * (src_arg as i32 + 1)]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; vpxor Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vmovss Rx(reg(out_reg)), [rcx + 4 * (src_arg as i32)]
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn grad_sin(v: Grad) -> Grad {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, grad_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_cos(f: Grad) -> Grad {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_tan(f: Grad) -> Grad {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_asin(f: Grad) -> Grad {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_acos(f: Grad) -> Grad {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_atan(f: Grad) -> Grad {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_exp(f: Grad) -> Grad {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_ln(f: Grad) -> Grad {
            f.ln()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ln);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovups Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpcmpeqw xmm0, xmm0, xmm0
            ; vpslld xmm0, xmm0, 31 // set the sign bit
            ; vpxor Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Store 0.0 to xmm0, for comparisons
            ; vpxor xmm0, xmm0, xmm0

            ; vcomiss Rx(reg(lhs_reg)), xmm0
            ; jb >N

            // Fallthrough: non-negative (or NaN) input
            ; vmovups Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >E

            ; N: // negative
            ; vpcmpeqw xmm0, xmm0, xmm0
            ; vpslld xmm0, xmm0, 31 // set the sign bit
            ; vpxor Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
            // Fallthrough to end

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        // d/dx 1/f(x) = -f'(x) / f(x)**2
        dynasm!(self.0.ops
            // Calculate xmm0[0] = f(x)**2
            ; movss xmm0, Rx(reg(lhs_reg))
            ; mulss xmm0, xmm0

            // Negate it
            ; mov eax, 0x80000000u32 as i32
            ; movd xmm1, eax
            ; pxor xmm0, xmm1

            // Set every element in xmm0 to -f(x)**2
            ; vbroadcastss xmm0, xmm0

            // Set every element in xmm0 to -f'(x) / f(x)**2
            ; vdivps xmm0, Rx(reg(lhs_reg)), xmm0

            // Compute the actual reciprocal into xmm1
            ; mov eax, 1.0f32.to_bits() as i32
            ; movd xmm1, eax
            ; divss xmm1, Rx(reg(lhs_reg))

            ; vmovups Rx(reg(out_reg)), xmm0
            ; movss Rx(reg(out_reg)), xmm1
        );
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        // d/dx sqrt(f(x)) = f'(x) / (2 * sqrt(f(x)))
        dynasm!(self.0.ops
            // Calculate xmm0[0] = f(x)**2
            ; vsqrtss xmm0, xmm0, Rx(reg(lhs_reg))

            // Multiply it by 2
            // TODO is this the best way to make 2.0?
            ; mov eax, 2.0f32.to_bits() as i32
            ; vmovd xmm1, eax
            ; vmulss xmm0, xmm0, xmm1

            // Set every element in xmm0 to 2 * sqrt(f(x))
            ; vbroadcastss xmm0, xmm0

            // Set every element in xmm0 to -f'(x) / f(x)**2
            ; vdivps xmm0, Rx(reg(lhs_reg)), xmm0

            // Compute the actual square root into xmm1
            ; vsqrtss xmm1, xmm1, Rx(reg(lhs_reg))

            ; vmovups Rx(reg(out_reg)), xmm0
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm1
        );
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        // d/dx f(x)**2 = 2 * f(x) * f'(x)
        dynasm!(self.0.ops
            ; mov eax, 2.0f32.to_bits() as i32
            ; vmovd xmm1, eax
            ; vbroadcastss xmm1, xmm1

            ; mov eax, 1.0f32.to_bits() as i32
            ; vmovd xmm0, eax
            ; movss xmm1, xmm0
            // At this point, xmm1 contains [1, 2, 2, 2]

            ; vbroadcastss xmm0, Rx(reg(lhs_reg))
            ; vmulps xmm0, xmm0, xmm1
            ; vmulps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        );
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vaddps Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vsubps Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // d/dx f(x) * g(x) = f'(x)*g(x) + f(x)*g'(x)
        dynasm!(self.0.ops
            ; vbroadcastss xmm1, Rx(reg(lhs_reg))
            ; vmulps xmm1, xmm1, Rx(reg(rhs_reg))
            ; vbroadcastss xmm2, Rx(reg(rhs_reg))
            ; vmulps xmm2, xmm2, Rx(reg(lhs_reg))
            ; vaddps xmm1, xmm1, xmm2

            ; vmulss xmm2, Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; vmovups Rx(reg(out_reg)), xmm1
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm2
        );
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // d/dx f(x) * g(x) = (f'(x)*g(x) - f(x)*g'(x)) / g(x)**2
        dynasm!(self.0.ops
            // f(x) * g'(x)
            ; vbroadcastss xmm1, Rx(reg(lhs_reg))
            ; vmulps xmm1, xmm1, Rx(reg(rhs_reg))

            // g(x) * f'(x)
            ; vbroadcastss xmm2, Rx(reg(rhs_reg))
            ; vmulps xmm2, xmm2, Rx(reg(lhs_reg))

            // f'(x)*g(x) - f(x)*g'(x)
            ; vsubps xmm1, xmm2, xmm1

            // g(x)**2
            ; vmulss xmm2, Rx(reg(rhs_reg)), Rx(reg(rhs_reg))
            ; vbroadcastss xmm2, xmm2

            // Do the division
            ; vdivps xmm1, xmm1, xmm2

            // Patch in the actual division result
            ; vdivss xmm2, Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; vmovups Rx(reg(out_reg)), xmm1
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm2
        );
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; ja >L

            // Fallthrough
            ; vmovups Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; jmp >E

            ; L:
            ; vmovups Rx(reg(out_reg)), Rx(reg(lhs_reg))
            // Fallthrough

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; ja >R

            // Fallthrough
            ; vmovups Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            ; R:
            ; vmovups Rx(reg(out_reg)), Rx(reg(rhs_reg))
            // Fallthrough

            ; O:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; vpxor Rx(IMM_REG), Rx(IMM_REG), Rx(IMM_REG)
            ; mov eax, imm_u32 as i32
            ; vmovd Rx(IMM_REG), eax
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Copy data from out_reg into the out array, then adjust it
            ; vmovups [r8], Rx(reg(out_reg))
            ; add r8, 16 // 4x float
            ; sub r9, 1
            ; jmp ->L

            // Finalization code, which happens after all evaluation is complete
            ; -> X:
            ; add rsp, self.0.mem_offset as i32
            ; pop rbp
            ; emms
            ; ret
        );

        self.0.ops.finalize()
    }
}

impl GradSliceAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "sysv64" fn(Grad) -> Grad,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up X/Y/Z pointers to caller-saved registers and the stack
            ; mov r12, rdi
            ; mov r13, rsi
            ; mov r14, rdx
            ; push rcx
            ; push r8
            ; push r9
            ; push r9 // alignment?

            // Back up register values to the stack
            ; sub rsp, 192
            ; vmovups [rsp], xmm4
            ; vmovups [rsp + 16], xmm5
            ; vmovups [rsp + 32], xmm6
            ; vmovups [rsp + 48], xmm7
            ; vmovups [rsp + 64], xmm8
            ; vmovups [rsp + 80], xmm9
            ; vmovups [rsp + 96], xmm10
            ; vmovups [rsp + 112], xmm11
            ; vmovups [rsp + 128], xmm12
            ; vmovups [rsp + 144], xmm13
            ; vmovups [rsp + 160], xmm14
            ; vmovups [rsp + 176], xmm15

            // call the function, packing the gradient into xmm0 + xmm1
            ; movsd xmm0, Rx(reg(arg_reg))
            ; vpshufd xmm1, Rx(reg(arg_reg)), 0b1110
            ; mov rdx, QWORD addr as _
            ; call rdx

            // Restore gradient registers
            ; vmovups xmm4, [rsp]
            ; vmovups xmm5, [rsp + 16]
            ; vmovups xmm6, [rsp + 32]
            ; vmovups xmm7, [rsp + 48]
            ; vmovups xmm8, [rsp + 64]
            ; vmovups xmm9, [rsp + 80]
            ; vmovups xmm10, [rsp + 96]
            ; vmovups xmm11, [rsp + 112]
            ; vmovups xmm12, [rsp + 128]
            ; vmovups xmm13, [rsp + 144]
            ; vmovups xmm14, [rsp + 160]
            ; vmovups xmm15, [rsp + 176]
            ; add rsp, 192

            // Restore X/Y/Z pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14

            // Collect the 4x floats into the out register
            ; vpunpcklqdq Rx(reg(out_reg)), xmm0, xmm1

            // Restore X/Y/Z pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14
            ; pop r9 // alignment
            ; pop r9
            ; pop r8
            ; pop rcx
        );
    }
}
