use crate::jit::{
    float_slice::FloatSliceAssembler, mmap::Mmap, reg, Assembler,
    AssemblerData, Error, IMM_REG, OFFSET, REGISTER_LIMIT,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub const SIMD_WIDTH: usize = 8;

/// Assembler for SIMD point-wise evaluation on `x86_64`
///
/// Arguments are passed as follows:
///
/// | Argument | Register | Type                       |
/// | ---------|----------|----------------------------|
/// | vars     | `rdi`    | `*const *const [f32; 8]`   |
/// | out      | `rsi`    | `*mut [f32; 8]`            |
/// | size     | `rdx`    | `u64`                      |
///
/// The arrays must be an even multiple of 8 floats, since we're using AVX2 and
/// 256-bit wide operations for everything.
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
/// | -0x28    | `r15`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      | ...          | Register spills live up here                |
/// | 0x220    | ...          |                                             |
/// |----------|--------------|---------------------------------------------|
/// | 0x200    | function in  | Stashed arguments for function calls        |
/// | 0x180    | function i/o | Inputs and outputs for function calls       |
/// |----------|--------------|---------------------------------------------|
/// | 0x160    | ymm15        | Caller-saved registers during functions     |
/// | 0x140    | ymm14        | calls are placed here, then restored        |
/// | 0x120    | ymm13        |                                             |
/// | 0x100    | ymm12        |                                             |
/// | 0xe0     | ymm11        |                                             |
/// | 0xc0     | ymm10        |                                             |
/// | 0xa0     | ymm9         |                                             |
/// | 0x80     | ymm8         |                                             |
/// | 0x60     | ymm7         |                                             |
/// | 0x40     | ymm6         |                                             |
/// | 0x20     | ymm5         |                                             |
/// | 0x00     | ymm4         |                                             |
/// ```
const STACK_SIZE_UPPER: usize = 0x28; // Positions relative to `rbp`
const STACK_SIZE_LOWER: usize = 0x220; // Positions relative to `rsp`

impl Assembler for FloatSliceAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
        );
        out.prepare_stack(slot_count, STACK_SIZE_UPPER + STACK_SIZE_LOWER);
        dynasm!(out.ops
            // TODO should there be a `vzeroupper` in here?

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
            ; vmovups Ry(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(dst_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        dynasm!(self.0.ops
            ; vmovups [rsp + sp_offset], Ry(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        let pos = 8 * (src_arg as i32);
        dynasm!(self.0.ops
            ; mov r8, [rdi + pos]   // read the *const float from the array
            ; add r8, rcx           // offset it by array position
            ; vmovups Ry(reg(out_reg)), [r8]
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_sin(f: f32) -> f32 {
            f.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_cos(f: f32) -> f32 {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_tan(f: f32) -> f32 {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_asin(f: f32) -> f32 {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_acos(f: f32) -> f32 {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_atan(f: f32) -> f32 {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_exp(f: f32) -> f32 {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_ln(f: f32) -> f32 {
            f.ln()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ln);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovups Ry(reg(out_reg)), Ry(reg(lhs_reg))
        );
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpcmpeqw ymm0, ymm0, ymm0
            ; vpslld ymm0, ymm0, 31 // set the sign bit
            ; vpxor Ry(reg(out_reg)), ymm0, Ry(reg(lhs_reg))
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpcmpeqw ymm0, ymm0, ymm0
            ; vpsrld ymm0, ymm0, 1 // everything but the sign bit
            ; vpand Ry(reg(out_reg)), ymm0, Ry(reg(lhs_reg))
        );
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Build [1.0 x 8] in ymm0
            ; vpcmpeqw ymm0, ymm0, ymm0
            ; vpslld ymm0, ymm0, 25
            ; vpsrld ymm0, ymm0, 2
            ; vdivps Ry(reg(out_reg)), ymm0, Ry(reg(lhs_reg))
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

    // TODO optimize these three functions
    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_floor(f: f32) -> f32 {
            f.floor()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_floor);
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_ceil(f: f32) -> f32 {
            f.ceil()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ceil);
    }
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_round(f: f32) -> f32 {
            f.round()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_round);
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
    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        extern "sysv64" fn float_atan2(y: f32, x: f32) -> f32 {
            y.atan2(x)
        }
        self.call_fn_binary(out_reg, lhs_reg, rhs_reg, float_atan2);
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a mask of NANs; conveniently, all 1s is a NAN
            ; vcmpunordps ymm1, Ry(reg(lhs_reg)), Ry(reg(lhs_reg))
            ; vcmpunordps ymm2, Ry(reg(rhs_reg)), Ry(reg(rhs_reg))
            ; vorps ymm1, ymm2, ymm1

            // Calculate the max, which ignores NANs
            ; vmaxps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))

            // Set the NAN bits
            ; vorps Ry(reg(out_reg)), Ry(reg(out_reg)), ymm1
        );
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a mask of NANs; conveniently, all 1s is a NAN
            ; vcmpunordps ymm1, Ry(reg(lhs_reg)), Ry(reg(lhs_reg))
            ; vcmpunordps ymm2, Ry(reg(rhs_reg)), Ry(reg(rhs_reg))
            ; vorps ymm1, ymm2, ymm1

            // Calculate the min, which ignores NANs
            ; vminps Ry(reg(out_reg)), Ry(reg(lhs_reg)), Ry(reg(rhs_reg))

            // Set the NAN bits
            // (note that we leave other bits unchanged, because it doesn't
            // matter here!)
            ; vorps Ry(reg(out_reg)), Ry(reg(out_reg)), ymm1
        );
    }
    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Take abs(rhs_reg)
            ; vpcmpeqw ymm2, ymm2, ymm2
            ; vpsrld ymm2, ymm2, 1 // everything but the sign bit
            ; vpand ymm1, ymm2, Ry(reg(rhs_reg))

            ; vdivps ymm2, Ry(reg(lhs_reg)), ymm1
            ; vroundps ymm2, ymm2, 0b1 // floor
            ; vmulps ymm2, ymm2, ymm1
            ; vsubps Ry(reg(out_reg)), Ry(reg(lhs_reg)), ymm2
        );
    }
    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        dynasm!(self.0.ops
            ; vxorps ymm1, ymm1, ymm1
            ; vcmpeqps ymm1, ymm1, Ry(reg(arg_reg))
            ; mov eax, 1f32.to_bits() as i32
            ; vmovd Rx(reg(out_reg)), eax
            ; vbroadcastss Ry(reg(out_reg)), Rx(reg(out_reg))
            ; vandpd Ry(reg(out_reg)), Ry(reg(out_reg)), ymm1
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build the (lhs == 0) mask in ymm1 and the opposite in ymm2
            ; vxorps ymm1, ymm1, ymm1
            ; vcmpeqps ymm1, ymm1, Ry(reg(lhs_reg))
            ; vpcmpeqd ymm2, ymm2, ymm2 // All 1s
            ; vxorpd ymm2, ymm1, ymm2 // 1 ^ b = !b, so this inverts ymm1

            ; vandpd ymm1, ymm1, Ry(reg(lhs_reg))
            ; vandpd ymm2, ymm2, Ry(reg(rhs_reg))
            ; vorpd Ry(reg(out_reg)), ymm1, ymm2
        );
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build the (lhs == 0) mask in ymm1 and the opposite in ymm2
            ; vxorps ymm1, ymm1, ymm1
            ; vcmpeqps ymm1, ymm1, Ry(reg(lhs_reg))
            ; vpcmpeqd ymm2, ymm2, ymm2 // All 1s
            ; vxorpd ymm2, ymm1, ymm2 // 1 ^ b = !b, so this inverts ymm1

            ; vandpd ymm1, ymm1, Ry(reg(rhs_reg))
            ; vandpd ymm2, ymm2, Ry(reg(lhs_reg))
            ; vorpd Ry(reg(out_reg)), ymm1, ymm2
        );
    }

    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Build a mask of NANs; conveniently, all 1s is a NAN
            ; vcmpunordps ymm1, Ry(reg(lhs_reg)), Ry(reg(lhs_reg))
            ; vcmpunordps ymm2, Ry(reg(rhs_reg)), Ry(reg(rhs_reg))
            ; vorps ymm1, ymm2, ymm1

            // Calculate the less-than mask in ymm2
            ; vcmpltps ymm2, Ry(reg(lhs_reg)), Ry(reg(rhs_reg))

            // Calculate the greater-than mask in ymm2
            ; vcmpgtps ymm3, Ry(reg(lhs_reg)), Ry(reg(rhs_reg))

            // Put [-1.0; N] into the output register
            ; mov eax, (-1f32).to_bits() as i32
            ; vmovd Rx(reg(out_reg)), eax
            ; vbroadcastss Ry(reg(out_reg)), Rx(reg(out_reg))

            // Apply the less-than mask to the [-1.0 x N] reg
            ; vandps Ry(reg(out_reg)), Ry(reg(out_reg)), ymm2

            // Build and apply [1.0 x N] & greater-than
            ; mov eax, 1f32.to_bits() as i32
            ; vmovd xmm2, eax
            ; vbroadcastss ymm2, xmm2
            ; vandps ymm2, ymm2, ymm3
            ; vorps Ry(reg(out_reg)), Ry(reg(out_reg)), ymm2

            // Set the NAN bits
            ; vorps Ry(reg(out_reg)), Ry(reg(out_reg)), ymm1
        );
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        dynasm!(self.0.ops
            ; mov eax, imm.to_bits() as i32
            ; vmovd Rx(IMM_REG), eax
            ; vbroadcastss Ry(IMM_REG), Rx(IMM_REG)
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Copy data from out_reg into the out array, then adjust it
            ; vmovups [rsi], Ry(reg(out_reg))
            ; add rsi, 32
            ; sub rdx, 8
            ; add rcx, 32
            ; jmp ->L

            // Finalization code, which happens after all evaluation is complete
            ; ->X:
            ; add rsp, self.0.mem_offset as i32
            ; pop rbp
            ; emms
            ; vzeroall
            ; ret
        );

        self.0.ops.finalize()
    }
}

impl FloatSliceAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "sysv64" fn(f32) -> f32,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up all of our pointers to the stack
            ; mov [rbp - 0x8], rdi
            ; mov [rbp - 0x10], rsi
            ; mov [rbp - 0x18], rdx
            ; mov [rbp - 0x20], rcx
            ; mov [rbp - 0x28], r15

            // Back up register values to the stack, saving all 128 bits
            ; vmovups [rsp], ymm4
            ; vmovups [rsp + 0x20], ymm5
            ; vmovups [rsp + 0x40], ymm6
            ; vmovups [rsp + 0x60], ymm7
            ; vmovups [rsp + 0x80], ymm8
            ; vmovups [rsp + 0xa0], ymm9
            ; vmovups [rsp + 0xc0], ymm10
            ; vmovups [rsp + 0xe0], ymm11
            ; vmovups [rsp + 0x100], ymm12
            ; vmovups [rsp + 0x120], ymm13
            ; vmovups [rsp + 0x140], ymm14
            ; vmovups [rsp + 0x160], ymm15

            // Put the function pointer into a caller-saved register
            ; mov r15, QWORD addr as _
            ; vmovups [rsp + 0x180], Ry(reg(arg_reg))

            ; movd xmm0, [rsp + 0x180]
            ; call r15
            ; movd [rsp + 0x180], xmm0
            ; movd xmm0, [rsp + 0x184]
            ; call r15
            ; movd [rsp + 0x184], xmm0
            ; movd xmm0, [rsp + 0x188]
            ; call r15
            ; movd [rsp + 0x188], xmm0
            ; movd xmm0, [rsp + 0x18c]
            ; call r15
            ; movd [rsp + 0x18c], xmm0
            ; movd xmm0, [rsp + 0x190]
            ; call r15
            ; movd [rsp + 0x190], xmm0
            ; movd xmm0, [rsp + 0x194]
            ; call r15
            ; movd [rsp + 0x194], xmm0
            ; movd xmm0, [rsp + 0x198]
            ; call r15
            ; movd [rsp + 0x198], xmm0
            ; movd xmm0, [rsp + 0x19c]
            ; call r15
            ; movd [rsp + 0x19c], xmm0

            // Restore float registers
            ; vmovups ymm4, [rsp]
            ; vmovups ymm5, [rsp + 0x20]
            ; vmovups ymm6, [rsp + 0x40]
            ; vmovups ymm7, [rsp + 0x60]
            ; vmovups ymm8, [rsp + 0x80]
            ; vmovups ymm9, [rsp + 0xa0]
            ; vmovups ymm10, [rsp + 0xc0]
            ; vmovups ymm11, [rsp + 0xe0]
            ; vmovups ymm12, [rsp + 0x100]
            ; vmovups ymm13, [rsp + 0x120]
            ; vmovups ymm14, [rsp + 0x140]
            ; vmovups ymm15, [rsp + 0x160]

            // Get the output value from the stack
            ; vmovups Ry(reg(out_reg)), [rsp + 0x180]

            // Restore pointers
            ; mov rdi, [rbp - 0x8]
            ; mov rsi, [rbp - 0x10]
            ; mov rdx, [rbp - 0x18]
            ; mov rcx, [rbp - 0x20]
            ; mov r15, [rbp - 0x28]
        );
    }
    fn call_fn_binary(
        &mut self,
        out_reg: u8,
        lhs_reg: u8,
        rhs_reg: u8,
        f: extern "sysv64" fn(f32, f32) -> f32,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up all of our pointers to the stack
            ; mov [rbp - 0x8], rdi
            ; mov [rbp - 0x10], rsi
            ; mov [rbp - 0x18], rdx
            ; mov [rbp - 0x20], rcx
            ; mov [rbp - 0x28], r15

            // Back up register values to the stack, saving all 128 bits
            ; vmovups [rsp], ymm4
            ; vmovups [rsp + 0x20], ymm5
            ; vmovups [rsp + 0x40], ymm6
            ; vmovups [rsp + 0x60], ymm7
            ; vmovups [rsp + 0x80], ymm8
            ; vmovups [rsp + 0xa0], ymm9
            ; vmovups [rsp + 0xc0], ymm10
            ; vmovups [rsp + 0xe0], ymm11
            ; vmovups [rsp + 0x100], ymm12
            ; vmovups [rsp + 0x120], ymm13
            ; vmovups [rsp + 0x140], ymm14
            ; vmovups [rsp + 0x160], ymm15

            // Put the function pointer into a caller-saved register
            ; mov r15, QWORD addr as _

            // Copy our input arguments to the stack for safe-keeping
            ; vmovups [rsp + 0x180], Ry(reg(lhs_reg))
            ; vmovups [rsp + 0x200], Ry(reg(rhs_reg))

            ; movd xmm0, [rsp + 0x180]
            ; movd xmm1, [rsp + 0x200]
            ; call r15
            ; movd [rsp + 0x180], xmm0
            ; movd xmm0, [rsp + 0x184]
            ; movd xmm1, [rsp + 0x204]
            ; call r15
            ; movd [rsp + 0x184], xmm0
            ; movd xmm0, [rsp + 0x188]
            ; movd xmm1, [rsp + 0x208]
            ; call r15
            ; movd [rsp + 0x188], xmm0
            ; movd xmm0, [rsp + 0x18c]
            ; movd xmm1, [rsp + 0x20c]
            ; call r15
            ; movd [rsp + 0x18c], xmm0
            ; movd xmm0, [rsp + 0x190]
            ; movd xmm1, [rsp + 0x210]
            ; call r15
            ; movd [rsp + 0x190], xmm0
            ; movd xmm0, [rsp + 0x194]
            ; movd xmm1, [rsp + 0x214]
            ; call r15
            ; movd [rsp + 0x194], xmm0
            ; movd xmm0, [rsp + 0x198]
            ; movd xmm1, [rsp + 0x218]
            ; call r15
            ; movd [rsp + 0x198], xmm0
            ; movd xmm0, [rsp + 0x19c]
            ; movd xmm1, [rsp + 0x21c]
            ; call r15
            ; movd [rsp + 0x19c], xmm0

            // Restore float registers
            ; vmovups ymm4, [rsp]
            ; vmovups ymm5, [rsp + 0x20]
            ; vmovups ymm6, [rsp + 0x40]
            ; vmovups ymm7, [rsp + 0x60]
            ; vmovups ymm8, [rsp + 0x80]
            ; vmovups ymm9, [rsp + 0xa0]
            ; vmovups ymm10, [rsp + 0xc0]
            ; vmovups ymm11, [rsp + 0xe0]
            ; vmovups ymm12, [rsp + 0x100]
            ; vmovups ymm13, [rsp + 0x120]
            ; vmovups ymm14, [rsp + 0x140]
            ; vmovups ymm15, [rsp + 0x160]

            // Get the output value from the stack
            ; vmovups Ry(reg(out_reg)), [rsp + 0x180]

            // Restore pointers
            ; mov rdi, [rbp - 0x8]
            ; mov rsi, [rbp - 0x10]
            ; mov rdx, [rbp - 0x18]
            ; mov rcx, [rbp - 0x20]
            ; mov r15, [rbp - 0x28]
        );
    }
}
