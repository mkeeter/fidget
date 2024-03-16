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
/// | Argument | Register | Type                |
/// | ---------|----------|---------------------|
/// | X        | `rdi`    | `*const [f32; 8]`   |
/// | Y        | `rsi`    | `*const [f32; 8]`   |
/// | Z        | `rdx`    | `*const [f32; 8]`   |
/// | vars     | `rcx`    | `*const f32`        |
/// | out      | `r8`     | `*mut [f32; 8]`     |
/// | size     | `r9`     | `u64`               |
///
/// The arrays (other than `vars`) must be an even multiple of 8 floats, since
/// we're using AVX2 and 256-bit wide operations for everything.  The `vars`
/// array contains single `f32` values, which are broadcast into SIMD registers
/// when they are used.
///
/// During evaluation, X, Y, and Z values are stored on the stack to keep
/// registers unoccupied.
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
/// | -0x28    | `r8`         |                                             |
/// | -0x30    | `r9`         |                                             |
/// | -0x38    | `r15`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | -0x40    | Z            | Inputs (as 8x floats)                       |
/// | -0x60    | Y            |                                             |
/// | -0x80    | X            |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
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
const STACK_SIZE_UPPER: usize = 0x80; // Positions relative to `rbp`
const STACK_SIZE_LOWER: usize = 0x200; // Positions relative to `rsp`

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

            // The loop returns here, and we check whether to keep looping
            ; ->L:

            ; test r9, r9
            ; jz ->X // jump to the exit if we're done, otherwise fallthrough

            // Copy from the input pointers into the stack
            ; vmovups ymm0, [rdi]
            ; vmovups [rbp - (STACK_SIZE_UPPER as i32)], ymm0
            ; add rdi, 32

            ; vmovups ymm0, [rsi]
            ; vmovups [rbp - (STACK_SIZE_UPPER as i32 - 32)], ymm0
            ; add rsi, 32

            ; vmovups ymm0, [rdx]
            ; vmovups [rbp - (STACK_SIZE_UPPER as i32 - 32 * 2)], ymm0
            ; add rdx, 32
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
        let pos = STACK_SIZE_UPPER as i32 - 32 * (src_arg as i32);
        dynasm!(self.0.ops
            ; vmovups Ry(reg(out_reg)), [rbp - pos]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), [rcx + 4 * (src_arg as i32)]
            ; vbroadcastss Ry(reg(out_reg)), Rx(reg(out_reg))
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
            ; vmovups [r8], Ry(reg(out_reg))
            ; add r8, 32
            ; sub r9, 8
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
            // Back up X/Y/Z pointers to the stack
            ; mov [rbp - 0x8], rdi
            ; mov [rbp - 0x10], rsi
            ; mov [rbp - 0x18], rdx
            ; mov [rbp - 0x20], rcx
            ; mov [rbp - 0x28], r8
            ; mov [rbp - 0x30], r9
            ; mov [rbp - 0x38], r15

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

            // Restore X/Y/Z pointers
            ; mov rdi, [rbp - 0x8]
            ; mov rsi, [rbp - 0x10]
            ; mov rdx, [rbp - 0x18]
            ; mov rcx, [rbp - 0x20]
            ; mov r8, [rbp - 0x28]
            ; mov r9, [rbp - 0x30]
            ; mov r15, [rbp - 0x38]
        );
    }
}
