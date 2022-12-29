//! Compilation down to native machine code
//!
//! ```
//! use fidget::{rhai::eval, jit, eval::Eval};
//!
//! let (sum, ctx) = eval("x + y").unwrap();
//! let tape = ctx.get_tape(sum).unwrap();
//!
//! // Generate machine code to execute the tape
//! let mut eval = jit::Eval::new_point_evaluator(tape);
//!
//! // This calls directly into that machine code!
//! assert_eq!(eval.eval(0.1, 0.3, 0.0, &[]).unwrap().0, 0.1 + 0.3);
//! ```

// # Notes for writing assembly in this module
// ## Working registers
// We dedicate 24 registers to tape data storage:
// - Floating point registers `s8-15` (callee-saved, but only the lower 64
//   bits)
// - Floating-point registers `s16-31` (caller-saved)
//
// This means that the input tape must be planned with a <= 24 register limit;
// any spills will live on the stack.
//
// Right now, we never call anything, so don't worry about saving stuff.
//
// ## Scratch registers
// Within a single operation, you'll often need to make use of scratch
// registers.  `s3` / `v3` is used when loading immediates, and should not be
// used as a scratch register (this is the `IMM_REG` constant).  `s4-7`/`v4-7`
// are all available, and are callee-saved.
//
// For general-purpose registers, `x9-15` (also called `w9-15`) are reasonable
// choices; they are caller-saved, so we can trash them at will.

mod mmap;

use dynasmrt::{dynasm, AssemblyOffset, DynasmApi};

use crate::{
    eval::{tape::Data as TapeData, Choice, Family},
    jit::mmap::Mmap,
    vm::Op,
};

/// Number of registers available when executing natively
///
/// We can use registers v8-v15 (callee saved) and v16-v31 (caller saved)
const REGISTER_LIMIT: u8 = 24;

/// Offset before the first useable register
const OFFSET: u8 = 8;

/// Register written to by `CopyImm`
///
/// `IMM_REG` is selected to avoid scratch registers used by other
/// functions, e.g. interval mul / min / max
const IMM_REG: u8 = 3;

/// Converts from a tape-local register to an AArch64 register
///
/// Tape-local registers are in the range `0..REGISTER_LIMIT`, while ARM
/// registers have an offset (based on calling convention).
///
/// This uses `wrapping_add` to support immediates, which are loaded into an ARM
/// register below `OFFSET` (which is "negative" from the perspective of this
/// function).
fn reg(r: u8) -> u32 {
    let out = r.wrapping_add(OFFSET) as u32;
    assert!(out < 32);
    out
}

const CHOICE_LEFT: u32 = Choice::Left as u32;
const CHOICE_RIGHT: u32 = Choice::Right as u32;
const CHOICE_BOTH: u32 = Choice::Both as u32;

trait AssemblerT {
    fn init(m: Mmap, slot_count: usize) -> Self;
    fn build_load(&mut self, dst_reg: u8, src_mem: u32);
    fn build_store(&mut self, dst_mem: u32, src_reg: u8);

    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8);
    fn build_var(&mut self, out_reg: u8, src_arg: u32);
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8);
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_fma(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    // Special-case functions for immediates.  In some cases, you can be more
    // efficient if you know that an argument is an immediate (for example, both
    // values in the interval will be the same, and it wlll have no gradients).
    fn build_add_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_add(out_reg, lhs_reg, imm);
    }
    fn build_sub_imm_reg(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_sub(out_reg, imm, arg);
    }
    fn build_sub_reg_imm(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_sub(out_reg, arg, imm);
    }
    fn build_mul_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_mul(out_reg, lhs_reg, imm);
    }
    fn build_fma_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_fma(out_reg, lhs_reg, imm);
    }

    /// Loads an immediate into a register, returning that register
    fn load_imm(&mut self, imm: f32) -> u8;

    fn finalize(self, out_reg: u8) -> Mmap;
}

/////////////////////////////////////////////////////////////////////////////////////////

struct AssemblerData<T> {
    ops: MmapAssembler,

    /// Current offset of the stack pointer, in bytes
    mem_offset: usize,

    _p: std::marker::PhantomData<*const T>,
}

impl<T> AssemblerData<T> {
    fn new(mmap: Mmap) -> Self {
        Self {
            ops: MmapAssembler::from(mmap),
            mem_offset: 0,
            _p: std::marker::PhantomData,
        }
    }
    fn prepare_stack(&mut self, slot_count: usize) {
        if slot_count < REGISTER_LIMIT as usize {
            return;
        }
        let stack_slots = slot_count - REGISTER_LIMIT as usize;
        let mem = (stack_slots + 1) * std::mem::size_of::<T>();

        // Round up to the nearest multiple of 16 bytes, for alignment
        self.mem_offset = ((mem + 15) / 16) * 16;
        assert!(self.mem_offset < 4096);
        dynasm!(self.ops
            ; sub sp, sp, #(self.mem_offset as u32)
        );
    }
    fn stack_pos(&self, slot: u32) -> u32 {
        assert!(slot >= REGISTER_LIMIT as u32);
        (slot - REGISTER_LIMIT as u32) * std::mem::size_of::<T>() as u32
    }
}

////////////////////////////////////////////////////////////////////////////////

struct MmapAssembler {
    mmap: Mmap,
    len: usize,
}

impl Extend<u8> for MmapAssembler {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = u8>,
    {
        for c in iter.into_iter() {
            self.push(c);
        }
    }
}

impl<'a> Extend<&'a u8> for MmapAssembler {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = &'a u8>,
    {
        for c in iter.into_iter() {
            self.push(*c);
        }
    }
}

impl DynasmApi for MmapAssembler {
    #[inline(always)]
    fn offset(&self) -> AssemblyOffset {
        AssemblyOffset(self.len)
    }

    #[inline(always)]
    fn push(&mut self, byte: u8) {
        // Resize to fit the next byte, if needed
        if self.len >= self.mmap.len() {
            self.expand_mmap();
        }
        self.mmap.write(self.len, byte);
        self.len += 1;
    }

    #[inline(always)]
    fn align(&mut self, alignment: usize, with: u8) {
        let offset = self.offset().0 % alignment;
        if offset != 0 {
            for _ in offset..alignment {
                self.push(with);
            }
        }
    }

    #[inline(always)]
    fn push_u32(&mut self, value: u32) {
        if self.len + 3 >= self.mmap.len() {
            self.expand_mmap();
        }
        for (i, b) in value.to_le_bytes().iter().enumerate() {
            self.mmap.write(self.len + i, *b);
        }
        self.len += 4;
    }
}

impl MmapAssembler {
    fn finalize(self) -> Mmap {
        self.mmap.flush(self.len);
        self.mmap
    }

    /// Doubles the size of the internal `Mmap` and copies over data
    fn expand_mmap(&mut self) {
        let mut next = Mmap::new(self.mmap.len() * 2).unwrap();
        next.as_mut_slice()[0..self.len].copy_from_slice(self.mmap.as_slice());
        std::mem::swap(&mut self.mmap, &mut next);
    }
}

impl From<Mmap> for MmapAssembler {
    fn from(mmap: Mmap) -> Self {
        Self { mmap, len: 0 }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

mod float_slice;
mod grad;
mod interval;
mod point;

////////////////////////////////////////////////////////////////////////////////

fn build_asm_fn<A: AssemblerT>(t: &TapeData) -> Mmap {
    build_asm_fn_with_storage::<A>(t, Mmap::new(1).unwrap())
}

fn build_asm_fn_with_storage<A: AssemblerT>(t: &TapeData, s: Mmap) -> Mmap {
    let _guard = Mmap::thread_mode_write();
    let mut asm = A::init(s, t.slot_count());

    for op in t.iter_asm() {
        match op {
            Op::Load(reg, mem) => {
                asm.build_load(reg, mem);
            }
            Op::Store(reg, mem) => {
                asm.build_store(mem, reg);
            }
            Op::Input(out, i) => {
                asm.build_input(out, i);
            }
            Op::Var(out, i) => {
                asm.build_var(out, i);
            }
            Op::NegReg(out, arg) => {
                asm.build_neg(out, arg);
            }
            Op::AbsReg(out, arg) => {
                asm.build_abs(out, arg);
            }
            Op::RecipReg(out, arg) => {
                asm.build_recip(out, arg);
            }
            Op::SqrtReg(out, arg) => {
                asm.build_sqrt(out, arg);
            }
            Op::CopyReg(out, arg) => {
                asm.build_copy(out, arg);
            }
            Op::SquareReg(out, arg) => {
                asm.build_square(out, arg);
            }
            Op::AddRegReg(out, lhs, rhs) => {
                asm.build_add(out, lhs, rhs);
            }
            Op::MulRegReg(out, lhs, rhs) => {
                asm.build_mul(out, lhs, rhs);
            }
            Op::DivRegReg(out, lhs, rhs) => {
                asm.build_div(out, lhs, rhs);
            }
            Op::SubRegReg(out, lhs, rhs) => {
                asm.build_sub(out, lhs, rhs);
            }
            Op::MinRegReg(out, lhs, rhs) => {
                asm.build_min(out, lhs, rhs);
            }
            Op::MaxRegReg(out, lhs, rhs) => {
                asm.build_max(out, lhs, rhs);
            }
            Op::AddRegImm(out, arg, imm) => {
                asm.build_add_imm(out, arg, imm);
            }
            Op::MulRegImm(out, arg, imm) => {
                asm.build_mul_imm(out, arg, imm);
            }
            Op::DivRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, arg, reg);
            }
            Op::DivImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, reg, arg);
            }
            Op::SubImmReg(out, arg, imm) => {
                asm.build_sub_imm_reg(out, arg, imm);
            }
            Op::SubRegImm(out, arg, imm) => {
                asm.build_sub_reg_imm(out, arg, imm);
            }
            Op::MinRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_min(out, arg, reg);
            }
            Op::MaxRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_max(out, arg, reg);
            }
            Op::CopyImm(out, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_copy(out, reg);
            }
        }
    }

    asm.finalize(0)
    // JIT execute mode is restored here when the _guard is dropped
}

/// JIT evaluator family
#[derive(Clone)]
pub enum Eval {}
impl Family for Eval {
    const REG_LIMIT: u8 = REGISTER_LIMIT;

    type IntervalEval = interval::JitIntervalEval;
    type FloatSliceEval = float_slice::JitFloatSliceEval;
    type GradEval = grad::JitGradEval;
    type PointEval = point::JitPointEval;

    fn tile_sizes_3d() -> &'static [usize] {
        &[64, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[128, 16]
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_tests!(Eval);
    crate::interval_tests!(Eval);
    crate::float_slice_tests!(Eval);
    crate::point_tests!(Eval);
}
