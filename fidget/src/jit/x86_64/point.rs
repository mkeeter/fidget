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
/// | X          | `xmm0`   | `f32`                 |
/// | Y          | `xmm1`   | `f32`                 |
/// | Z          | `xmm2`   | `f32`                 |
/// | `vars`     | `rdi`    | `*const f32` (array)  |
/// | `choices`  | `rsi`    | `*mut u8` (array)     |
/// | `simplify` | `rdx`    | `*mut u8` (single)    |
///
/// X, Y, and Z are stored on the stack during code execution, to free up those
/// registers as scratch values.
impl Assembler for PointAssembler {
    type Data = f32;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; vzeroupper
            // Put X/Y/Z on the stack so we can use those registers
            ; vmovss [rbp - 4], xmm0
            ; vmovss [rbp - 8], xmm1
            ; vmovss [rbp - 12], xmm2
        );
        out.prepare_stack(slot_count, 4);
        Self(out)
    }

    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(src_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovss Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = self.0.stack_pos(dst_mem).try_into().unwrap();
        dynasm!(self.0.ops
            ; vmovss [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        dynasm!(self.0.ops
            // Pull X/Y/Z from the stack, where they've been placed by init()
            ; vmovss Rx(reg(out_reg)), [rbp - 4 * (src_arg as i32 + 1)]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), [rdi + 4 * (src_arg as i32)]
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
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; mov eax, imm_u32 as i32
            ; vmovd Rx(IMM_REG), eax
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        dynasm!(self.0.ops
            // Prepare our return value
            ; vmovss xmm0, xmm0, Rx(reg(out_reg))
            ; add rsp, self.0.mem_offset as i32
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbp
            ; emms
            ; ret
        );
        self.0.ops.finalize()
    }
}

impl PointAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "sysv64" fn(f32) -> f32,
    ) {
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up X/Y/Z pointers to caller-saved registers
            ; mov r12, rdi
            ; mov r13, rsi
            ; mov r14, rdx

            // Back up X/Y/Z values to the stack
            ; sub rsp, 48
            ; movss [rsp], xmm4
            ; movss [rsp + 4], xmm5
            ; movss [rsp + 8], xmm6
            ; movss [rsp + 12], xmm7
            ; movss [rsp + 16], xmm8
            ; movss [rsp + 20], xmm9
            ; movss [rsp + 24], xmm10
            ; movss [rsp + 28], xmm11
            ; movss [rsp + 32], xmm12
            ; movss [rsp + 36], xmm13
            ; movss [rsp + 40], xmm14
            ; movss [rsp + 44], xmm15

            // call the function
            ; movss xmm0, Rx(reg(arg_reg))
            ; mov rdx, QWORD addr as _
            ; call rdx

            // Restore float registers
            ; movss xmm4, [rsp]
            ; movss xmm5, [rsp + 4]
            ; movss xmm6, [rsp + 8]
            ; movss xmm7, [rsp + 12]
            ; movss xmm8, [rsp + 16]
            ; movss xmm9, [rsp + 20]
            ; movss xmm10, [rsp + 24]
            ; movss xmm11, [rsp + 28]
            ; movss xmm12, [rsp + 32]
            ; movss xmm13, [rsp + 36]
            ; movss xmm14, [rsp + 40]
            ; movss xmm15, [rsp + 44]
            ; add rsp, 48

            // Restore X/Y/Z pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14

            ; movss Rx(reg(out_reg)), xmm0
        );
    }
}
