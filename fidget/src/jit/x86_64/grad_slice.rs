use crate::{
    jit::{
        grad_slice::GradSliceAssembler, mmap::Mmap, reg, Assembler,
        AssemblerData, IMM_REG, OFFSET, REGISTER_LIMIT,
    },
    types::Grad,
    Error,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation for the gradient slice assembler on `x86_64`
///
/// Registers as pased in as follows:
///
/// | Variable   | Register | Type                   |
/// |------------|----------|------------------------|
/// | `vars`     | `rdi`    | `*mut *const [f32; 4]` |
/// | `out`      | `rsi`    | `*mut [f32; 4]`        |
/// | `count`    | `rdx`    | `u64`                  |
///
/// During evaluation, `rcx` is used to track offset within `vars`.
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0x00     | `rbp`        | Previous value for base pointer             |
/// |----------|--------------|---------------------------------------------|
/// | -0x08    | `rdi`        | During functions calls, we use these        |
/// | -0x10    | `rsi`        | as temporary storage so must preserve their |
/// | -0x18    | `rdx`        | previous values on the stack                |
/// | -0x20    | `rcx`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
/// | 0xb0     | xmm15        | Caller-saved registers during functions     |
/// | 0xa0     | xmm14        | calls are placed here, then restored        |
/// | 0x90     | xmm13        |                                             |
/// | 0x80     | xmm12        |                                             |
/// | 0x70     | xmm11        |                                             |
/// | 0x60     | xmm10        |                                             |
/// | 0x50     | xmm9         |                                             |
/// | 0x40     | xmm8         |                                             |
/// | 0x30     | xmm7         |                                             |
/// | 0x20     | xmm6         |                                             |
/// | 0x10     | xmm5         |                                             |
/// | 0x00     | xmm4         |                                             |
/// ```
const STACK_SIZE_UPPER: usize = 0x20; // Positions relative to `rbp`
const STACK_SIZE_LOWER: usize = 0xc0; // Positions relative to `rsp`

impl Assembler for GradSliceAssembler {
    type Data = Grad;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
        );
        out.prepare_stack(slot_count, STACK_SIZE_UPPER + STACK_SIZE_LOWER);
        dynasm!(out.ops
            ; xor rcx, rcx // set the array offset (rcx) to 0

            // The loop returns here, and we check whether to keep looping
            ; ->L:

            ; test rdx, rdx
            ; jz ->X // jump to the exit if we're done, otherwise fallthrough
        );
        Self(out)
    }
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(src_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        dynasm!(self.0.ops
            ; vmovups Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(dst_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        // XXX could we use vmovaps here instead?
        dynasm!(self.0.ops
            ; vmovups [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        let pos = 8 * (src_arg as i32); // offset within the pointer array
        dynasm!(self.0.ops
            ; mov r8, [rdi + pos]   // read the *const float from the array
            ; add r8, rcx           // offset it by array position
            ; vmovaps Rx(reg(out_reg)), [r8]
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

    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vroundss xmm1, Rx(reg(lhs_reg)), Rx(reg(lhs_reg)), 1
            ; vpxor Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; movss Rx(reg(out_reg)), xmm1
        );
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vroundss xmm1, Rx(reg(lhs_reg)), Rx(reg(lhs_reg)), 2
            ; vpxor Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; movss Rx(reg(out_reg)), xmm1
        );
    }
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8) {
        // Shenanigans figured through Godbolt
        dynasm!(self.0.ops
            ; mov eax, 0x80000000u32 as i32
            ; vmovd xmm1, eax
            ; vandps  xmm1, xmm1, Rx(reg(lhs_reg))
            ; mov eax, 0x3effffffu32 as i32
            ; vmovd xmm2, eax
            ; vorps  xmm1, xmm1, xmm2
            ; vaddss Rx(reg(out_reg)), xmm1, Rx(reg(lhs_reg))
            ; vroundss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg)), 3
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

    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "sysv64" fn grad_atan2(y: Grad, x: Grad) -> Grad {
            y.atan2(x)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, grad_atan2);
    }

    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N // Parity flag is set if result is NAN
            ; ja >L

            // Fallthrough
            ; vmovups Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; jmp >E

            ; N:
            ; vpxor Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vcmpeqss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
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
            ; jp >N // Parity flag is set if result is NAN
            ; ja >R

            // Fallthrough
            ; vmovups Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            ; N:
            ; vpxor Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vcmpeqss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; jmp >O

            ; R:
            ; vmovups Rx(reg(out_reg)), Rx(reg(rhs_reg))
            // Fallthrough

            ; O:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "sysv64" fn grad_modulo(lhs: Grad, rhs: Grad) -> Grad {
            lhs.rem_euclid(rhs)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, grad_modulo);
    }
    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        let i = self.load_imm(1.0);
        dynasm!(self.0.ops
            ; vpxor xmm1, xmm1, xmm1
            ; vcmpeqss xmm1, Rx(reg(arg_reg)), xmm1
            ; vandps Rx(reg(out_reg)), xmm1, Rx(reg(i))
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpxor xmm1, xmm1, xmm1
            ; vcmpeqss xmm1, Rx(reg(lhs_reg)), xmm1
            ; vbroadcastss xmm1, xmm1
            ; vpcmpeqd xmm2, xmm2, xmm2
            ; vxorpd xmm2, xmm1, xmm2 // 1 ^ b = !b, so this inverts xmm1

            ; vandpd xmm1, xmm1, Rx(reg(lhs_reg))
            ; vandpd xmm2, xmm2, Rx(reg(rhs_reg))
            ; vorpd Rx(reg(out_reg)), xmm1, xmm2
        );
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpxor xmm1, xmm1, xmm1
            ; vcmpeqss xmm1, Rx(reg(lhs_reg)), xmm1
            ; vbroadcastss xmm1, xmm1
            ; vpcmpeqd xmm2, xmm2, xmm2
            ; vxorpd xmm2, xmm1, xmm2 // 1 ^ b = !b, so this inverts xmm1

            ; vandpd xmm1, xmm1, Rx(reg(rhs_reg))
            ; vandpd xmm2, xmm2, Rx(reg(lhs_reg))
            ; vorpd Rx(reg(out_reg)), xmm1, xmm2
        );
    }
    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N
            ; ja >R
            ; jb >L

            // Fall-through for equal
            ; xor eax, eax // set eax to 0u32, which is also 0f32
            ; vmovd Rx(reg(out_reg)), eax
            ; jmp >O

            // Less than
            ; L:
            ; mov eax, (-1f32).to_bits() as i32
            ; vmovd Rx(reg(out_reg)), eax
            ; jmp >O

            ; N:
            // TODO: this can't be the best way to make a NAN
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jmp >O

            ; R:
            ; mov eax, 1f32.to_bits() as i32
            ; vmovd Rx(reg(out_reg)), eax
            // fallthrough to out

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
            ; vmovups [rsi], Rx(reg(out_reg))
            ; add rsi, 16 // 4x float
            ; sub rdx, 1 // we process one element at a time
            ; add rcx, 16 // input is array is Grad (f32 x 4)
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
            // Back up X/Y/Z pointers to the stack
            ; mov [rbp - 0x8], rdi
            ; mov [rbp - 0x10], rsi
            ; mov [rbp - 0x18], rdx
            ; mov [rbp - 0x20], rcx

            // Back up register values to the stack, saving all 128 bits
            ; vmovups [rsp], xmm4
            ; vmovups [rsp + 0x10], xmm5
            ; vmovups [rsp + 0x20], xmm6
            ; vmovups [rsp + 0x30], xmm7
            ; vmovups [rsp + 0x40], xmm8
            ; vmovups [rsp + 0x50], xmm9
            ; vmovups [rsp + 0x60], xmm10
            ; vmovups [rsp + 0x70], xmm11
            ; vmovups [rsp + 0x80], xmm12
            ; vmovups [rsp + 0x90], xmm13
            ; vmovups [rsp + 0xa0], xmm14
            ; vmovups [rsp + 0xb0], xmm15

            // call the function, packing the gradient into xmm0 + xmm1
            ; movsd xmm0, Rx(reg(arg_reg))
            ; vpshufd xmm1, Rx(reg(arg_reg)), 0b1110
            ; mov rdx, QWORD addr as _
            ; call rdx

            // Restore gradient registers
            ; vmovups xmm4, [rsp]
            ; vmovups xmm5, [rsp + 0x10]
            ; vmovups xmm6, [rsp + 0x20]
            ; vmovups xmm7, [rsp + 0x30]
            ; vmovups xmm8, [rsp + 0x40]
            ; vmovups xmm9, [rsp + 0x50]
            ; vmovups xmm10, [rsp + 0x60]
            ; vmovups xmm11, [rsp + 0x70]
            ; vmovups xmm12, [rsp + 0x80]
            ; vmovups xmm13, [rsp + 0x90]
            ; vmovups xmm14, [rsp + 0xa0]
            ; vmovups xmm15, [rsp + 0xb0]

            // Restore X/Y/Z pointers
            ; mov rdi, [rbp - 0x8]
            ; mov rsi, [rbp - 0x10]
            ; mov rdx, [rbp - 0x18]
            ; mov rcx, [rbp - 0x20]

            // Collect the 4x floats into the out register
            ; vpunpcklqdq Rx(reg(out_reg)), xmm0, xmm1
        );
    }

    fn call_fn_binary(
        &mut self,
        out_reg: u8,
        lhs_reg: u8,
        rhs_reg: u8,
        f: extern "sysv64" fn(Grad, Grad) -> Grad,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up X/Y/Z pointers to the stack
            ; mov [rbp - 0x8], rdi
            ; mov [rbp - 0x10], rsi
            ; mov [rbp - 0x18], rdx
            ; mov [rbp - 0x20], rcx
            ; mov [rbp - 0x28], r8

            // Back up register values to the stack, saving all 128 bits
            ; vmovups [rsp], xmm4
            ; vmovups [rsp + 0x10], xmm5
            ; vmovups [rsp + 0x20], xmm6
            ; vmovups [rsp + 0x30], xmm7
            ; vmovups [rsp + 0x40], xmm8
            ; vmovups [rsp + 0x50], xmm9
            ; vmovups [rsp + 0x60], xmm10
            ; vmovups [rsp + 0x70], xmm11
            ; vmovups [rsp + 0x80], xmm12
            ; vmovups [rsp + 0x90], xmm13
            ; vmovups [rsp + 0xa0], xmm14
            ; vmovups [rsp + 0xb0], xmm15

            // Call the function, packing the gradient into xmm0 + xmm1
            // Note that we load xmm0 last, because it could be one of our
            // arguments if we're using IMM_REG
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 0b1110
            ; movsd xmm2, Rx(reg(rhs_reg))
            ; vpshufd xmm3, Rx(reg(rhs_reg)), 0b1110
            ; movsd xmm0, Rx(reg(lhs_reg))
            ; mov rdx, QWORD addr as _
            ; call rdx

            // Restore gradient registers
            ; vmovups xmm4, [rsp]
            ; vmovups xmm5, [rsp + 0x10]
            ; vmovups xmm6, [rsp + 0x20]
            ; vmovups xmm7, [rsp + 0x30]
            ; vmovups xmm8, [rsp + 0x40]
            ; vmovups xmm9, [rsp + 0x50]
            ; vmovups xmm10, [rsp + 0x60]
            ; vmovups xmm11, [rsp + 0x70]
            ; vmovups xmm12, [rsp + 0x80]
            ; vmovups xmm13, [rsp + 0x90]
            ; vmovups xmm14, [rsp + 0xa0]
            ; vmovups xmm15, [rsp + 0xb0]

            // Restore X/Y/Z pointers
            ; mov rdi, [rbp - 0x8]
            ; mov rsi, [rbp - 0x10]
            ; mov rdx, [rbp - 0x18]
            ; mov rcx, [rbp - 0x20]
            ; mov r8, [rbp - 0x28]

            // Collect the 4x floats into the out register
            ; vpunpcklqdq Rx(reg(out_reg)), xmm0, xmm1
        );
    }
}
