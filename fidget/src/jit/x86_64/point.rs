use crate::{
    jit::{
        mmap::Mmap, point::PointAssembler, reg, Assembler, AssemblerData,
        CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT, IMM_REG, OFFSET,
        REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation of the single-point assembler on `x86_64`
///
/// Registers are passed in as follows:
///
/// | Variable   | Register | Type                  |
/// |------------|----------|-----------------------|
/// | `vars`     | `rdi`    | `*const f32`          |
/// | `choices`  | `rsi`    | `*mut u8` (array)     |
/// | `simplify` | `rdx`    | `*mut u8` (single)    |
/// | `output`   | `rcx`    | `*mut f32` (array)    |
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0x00     | `rbp`        | Previous value for base pointer             |
/// |----------|--------------|---------------------------------------------|
/// | -0x08    | `r12`        | During functions calls, we use these        |
/// | -0x10    | `r13`        | as temporary storage so must preserve their |
/// | -0x18    | `r14`        | previous values on the stack                |
/// | -0x20    | `r15`        |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
/// | 0x2c     | xmm15        | Caller-saved registers during functions     |
/// | 0x28     | xmm14        | calls are placed here, then restored        |
/// | 0x24     | xmm13        |                                             |
/// | 0x20     | xmm12        |                                             |
/// | 0x1c     | xmm11        |                                             |
/// | 0x18     | xmm10        |                                             |
/// | 0x14     | xmm9         |                                             |
/// | 0x10     | xmm8         |                                             |
/// | 0x0c     | xmm7         |                                             |
/// | 0x08     | xmm6         |                                             |
/// | 0x04     | xmm5         |                                             |
/// | 0x00     | xmm4         |                                             |
/// ```
const STACK_SIZE_UPPER: usize = 0x20; // Positions relative to `rbp`
const STACK_SIZE_LOWER: usize = 0x30; // Positions relative to `rsp`

impl Assembler for PointAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
        );
        out.prepare_stack(slot_count, STACK_SIZE_UPPER + STACK_SIZE_LOWER);
        dynasm!(out.ops
            ; vzeroupper
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
            ; vmovss Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(dst_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        dynasm!(self.0.ops
            ; vmovss [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u32) {
        let pos = 4 * i32::try_from(src_arg).unwrap();
        dynasm!(self.0.ops
            // Pull the input from the rdi array
            ; vmovss Rx(reg(out_reg)), [rdi + pos]
        );
    }
    fn build_output(&mut self, arg_reg: u8, out_index: u32) {
        let pos = 4 * i32::try_from(out_index).unwrap();
        dynasm!(self.0.ops
            ; vmovss [rcx + pos], Rx(reg(arg_reg))
        );
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn point_sin(v: f32) -> f32 {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, point_sin);
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
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        // Flip the sign bit in the float
        dynasm!(self.0.ops
            // TODO: build this in xmm0 directly
            ; mov eax, 0x80000000u32 as i32
            ; movd xmm0, eax
            ; vxorps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        // Clear the sign bit in the float
        dynasm!(self.0.ops
            ; mov eax, 0x7fffffffu32 as i32
            ; vmovd xmm0, eax
            ; vandps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
        )
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        let imm = self.load_imm(1.0);
        dynasm!(self.0.ops
            ; divss Rx(reg(imm)), Rx(reg(lhs_reg))
            ; movss Rx(reg(out_reg)), Rx(reg(imm))
        );
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; sqrtss Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(lhs_reg))
        );
    }

    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vroundss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(lhs_reg)), 1
        );
    }
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vroundss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(lhs_reg)), 2
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
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        )
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vsubss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmulss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vdivss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
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
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N
            ; ja >L
            ; jb >R

            // Fallthrough for equal, so just copy to the output register
            ; or [rsi], CHOICE_BOTH as i8
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            // Fallthrough for NaN, which are !=; do a float addition to
            // propagate it to the output register.
            ; N:
            ; or [rsi], CHOICE_BOTH as i8
            // TODO: this can't be the best way to make a NAN
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jmp >O

            ; L:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or [rsi], CHOICE_LEFT as i8
            ; or [rdx], 1
            ; jmp >O

            ; R:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or [rsi], CHOICE_RIGHT as i8
            ; or [rdx], 1
            // fallthrough to out

            ; O:
        );
        self.0.ops.commit_local().unwrap()
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vcomiss Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jp >N
            ; ja >R
            ; jb >L

            // Fallthrough for equal, so just copy to the output register
            ; or [rsi], CHOICE_BOTH as i8
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >O

            ; N:
            ; or [rsi], CHOICE_BOTH as i8
            // TODO: this can't be the best way to make a NAN
            ; vaddss Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; jmp >O

            ; L:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or [rsi], CHOICE_LEFT as i8
            ; or [rdx], 1
            ; jmp >O

            ; R:
            ; vmovss Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or [rsi], CHOICE_RIGHT as i8
            ; or [rdx], 1
            // fallthrough to out

            ; O:
        );
        self.0.ops.commit_local().unwrap()
    }
    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Take abs(rhs_reg)
            ; mov eax, 0x7fffffffu32 as i32
            ; vmovd xmm2, eax
            ; vandps xmm1, xmm2, Rx(reg(rhs_reg))

            ; vdivss xmm2, Rx(reg(lhs_reg)), xmm1
            ; vroundss xmm2, xmm2, xmm2, 0b1 // floor
            ; vmulss xmm2, xmm2, xmm1
            ; vsubss Rx(reg(out_reg)), Rx(reg(lhs_reg)), xmm2
        );
    }
    fn build_not(&mut self, out_reg: u8, arg_reg: u8) {
        dynasm!(self.0.ops
            ; vxorps   xmm1, xmm1, xmm1
            ; vcmpeqss xmm1, xmm1, Rx(reg(arg_reg))
            ; mov eax, 1f32.to_bits() as i32
            ; vmovd Rx(reg(out_reg)), eax
            ; vandpd Rx(reg(out_reg)), Rx(reg(out_reg)), xmm1
        );
    }
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Based on Godbolt, so perhaps less readable than usual
            ; vmovaps xmm1, Rx(reg(rhs_reg))
            ; vxorps xmm2, xmm2, xmm2
            ; vucomiss xmm2, Rx(reg(lhs_reg))
            ; setnp r8b
            ; sete al
            ; jne >E
            ; jp >E
            ; vmovaps xmm1, Rx(reg(lhs_reg))

            ; E:
            ; and al, r8b
            ; mov r8b, 2
            ; sub r8b, al
            ; or [rsi], r8b // write the choice flag, based on condition flags
            ; or [rdx], 1 // write the simplify bit
            ; movaps Rx(reg(out_reg)), xmm1
        );
        self.0.ops.commit_local().unwrap()
    }
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            // Based on Godbolt, so perhaps less readable than usual
            ; vmovaps xmm1, Rx(reg(lhs_reg))
            ; vxorps xmm2, xmm2, xmm2
            ; vucomiss xmm2, Rx(reg(lhs_reg))
            ; setnp r8b
            ; sete al
            ; jne >E
            ; jp >E
            ; vmovaps xmm1, Rx(reg(rhs_reg))

            ; E:
            ; and al, r8b
            ; inc al
            ; or [rsi], al // write the choice flag, based on condition flags
            ; or [rdx], 1 // write the simplify bit
            ; movaps Rx(reg(out_reg)), xmm1
        );
        self.0.ops.commit_local().unwrap()
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
        self.0.ops.commit_local().unwrap()
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; mov eax, imm_u32 as i32
            ; vmovd Rx(IMM_REG), eax
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self) -> Result<Mmap, Error> {
        if self.0.saved_callee_regs {
            dynasm!(self.0.ops
                ; mov r12, [rbp - 0x8]
                ; mov r13, [rbp - 0x10]
                ; mov r14, [rbp - 0x18]
                ; mov r15, [rbp - 0x20]
            );
        }
        self.0.finalize()
    }
}

impl PointAssembler {
    fn ensure_callee_regs_saved(&mut self) {
        // Back up a few callee-saved registers that we're about to use
        if !self.0.saved_callee_regs {
            dynasm!(self.0.ops
                ; mov [rbp - 0x8], r12
                ; mov [rbp - 0x10], r13
                ; mov [rbp - 0x18], r14
                ; mov [rbp - 0x20], r15
            );
            self.0.saved_callee_regs = true
        }
    }
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "sysv64" fn(f32) -> f32,
    ) {
        self.ensure_callee_regs_saved();
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up pointers to caller-saved registers
            ; mov r12, rdi
            ; mov r13, rsi
            ; mov r14, rdx
            ; mov r15, rcx

            // Back up all register values to the stack
            ; movss [rsp], xmm4
            ; movss [rsp + 0x4], xmm5
            ; movss [rsp + 0x8], xmm6
            ; movss [rsp + 0xc], xmm7
            ; movss [rsp + 0x10], xmm8
            ; movss [rsp + 0x14], xmm9
            ; movss [rsp + 0x18], xmm10
            ; movss [rsp + 0x1c], xmm11
            ; movss [rsp + 0x20], xmm12
            ; movss [rsp + 0x24], xmm13
            ; movss [rsp + 0x28], xmm14
            ; movss [rsp + 0x2c], xmm15

            // call the function
            ; movss xmm0, Rx(reg(arg_reg))
            ; mov rsi, QWORD addr as _
            ; call rsi

            // Restore float registers
            ; movss xmm4, [rsp]
            ; movss xmm5, [rsp + 0x4]
            ; movss xmm6, [rsp + 0x8]
            ; movss xmm7, [rsp + 0xc]
            ; movss xmm8, [rsp + 0x10]
            ; movss xmm9, [rsp + 0x14]
            ; movss xmm10, [rsp + 0x18]
            ; movss xmm11, [rsp + 0x1c]
            ; movss xmm12, [rsp + 0x20]
            ; movss xmm13, [rsp + 0x24]
            ; movss xmm14, [rsp + 0x28]
            ; movss xmm15, [rsp + 0x2c]

            // Restore pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14
            ; mov rcx, r15

            ; movss Rx(reg(out_reg)), xmm0
        );
    }
    fn call_fn_binary(
        &mut self,
        out_reg: u8,
        lhs_reg: u8,
        rhs_reg: u8,
        f: extern "sysv64" fn(f32, f32) -> f32,
    ) {
        self.ensure_callee_regs_saved();
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up pointers to caller-saved registers
            ; mov r12, rdi
            ; mov r13, rsi
            ; mov r14, rdx
            ; mov r15, rcx

            // Back up all register values to the stack
            ; movss [rsp], xmm4
            ; movss [rsp + 0x4], xmm5
            ; movss [rsp + 0x8], xmm6
            ; movss [rsp + 0xc], xmm7
            ; movss [rsp + 0x10], xmm8
            ; movss [rsp + 0x14], xmm9
            ; movss [rsp + 0x18], xmm10
            ; movss [rsp + 0x1c], xmm11
            ; movss [rsp + 0x20], xmm12
            ; movss [rsp + 0x24], xmm13
            ; movss [rsp + 0x28], xmm14
            ; movss [rsp + 0x2c], xmm15

            // call the function.  Note the ordering here: xmm0 could be LHS /
            // RHS if we're doing a call with an immediate, so we overwrite it
            // last.
            ; movss xmm1, Rx(reg(rhs_reg))
            ; movss xmm0, Rx(reg(lhs_reg))
            ; mov rsi, QWORD addr as _
            ; call rsi

            // Restore float registers
            ; movss xmm4, [rsp]
            ; movss xmm5, [rsp + 0x4]
            ; movss xmm6, [rsp + 0x8]
            ; movss xmm7, [rsp + 0xc]
            ; movss xmm8, [rsp + 0x10]
            ; movss xmm9, [rsp + 0x14]
            ; movss xmm10, [rsp + 0x18]
            ; movss xmm11, [rsp + 0x1c]
            ; movss xmm12, [rsp + 0x20]
            ; movss xmm13, [rsp + 0x24]
            ; movss xmm14, [rsp + 0x28]
            ; movss xmm15, [rsp + 0x2c]

            // Restore pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14
            ; mov rcx, r15

            ; movss Rx(reg(out_reg)), xmm0
        );
    }
}
