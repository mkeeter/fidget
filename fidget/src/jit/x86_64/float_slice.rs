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
impl Assembler for FloatSliceAssembler {
    type Data = f32;

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
            ; jnz >B

            // Finalization code, which happens after all evaluation is complete
            ; add rsp, out.mem_offset as i32
            ; pop rbp
            ; emms
            ; vzeroall
            ; ret

            ; B:
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
        Self(out)
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
            ; vmovss Rx(reg(out_reg)), [rcx + 4 * (src_arg as i32)]
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
        );

        self.0.ops.finalize()
    }
}
