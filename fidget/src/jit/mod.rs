//! Compilation down to native machine code
//!
//! Users are unlikely to use anything in this module other than [`Eval`](Eval),
//! which is a [`Family`](Family) of JIT evaluators.
//!
//! ```
//! use fidget::{rhai::eval, jit};
//!
//! let (sum, ctx) = eval("x + y")?;
//! let tape = ctx.get_tape::<jit::Eval>(sum)?;
//!
//! // Generate machine code to execute the tape
//! let mut eval = tape.new_point_evaluator();
//!
//! // This calls directly into that machine code!
//! assert_eq!(eval.eval(0.1, 0.3, 0.0, &[])?.0, 0.1 + 0.3);
//! # Ok::<(), fidget::Error>(())
//! ```

use crate::{
    context::{BinaryOpcode, UnaryOpcode},
    eval::{
        bulk::BulkEvaluator, tape::Data as TapeData, tracing::TracingEvaluator,
        Choice, EvaluatorStorage, Family, Tape,
    },
    jit::mmap::Mmap,
    tape::Op,
};
use dynasmrt::{dynasm, AssemblyOffset, DynasmApi};
use std::sync::Arc;

mod mmap;

// Evaluators
mod float_slice;
mod grad_slice;
mod interval;
mod point;

// We allow `cargo doc` to build on x86, so that docs.rs includes documentation
// for the JIT module; however, everything is stubbed out.
#[cfg(not(any(doc, target_arch = "aarch64")))]
compile_error!(
    "The `jit` module only builds on `aarch64`; \
    please disable the `jit` feature"
);

#[cfg(not(any(doc, target_os = "macos")))]
compile_error!(
    "The `jit` module only builds on macOS; \
    please disable the `jit` feature"
);

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

/// Trait for generating machine assembly
///
/// This is public because it's used to parameterize various other types, but
/// shouldn't be used by clients; indeed, there are no public implementors of
/// this trait.
///
/// # Notes for writing assembly in this module
/// ## Working registers
/// We dedicate 24 registers to tape data storage:
/// - Floating point registers `s8-15` (callee-saved, but only the lower 64
///   bits)
/// - Floating-point registers `s16-31` (caller-saved)
///
/// This means that the input tape must be planned with a <= 24 register limit;
/// any spills will live on the stack.
///
/// Right now, we never call anything, so don't worry about saving stuff.
///
/// ## Scratch registers
/// Within a single operation, you'll often need to make use of scratch
/// registers.  `s3` / `v3` is used when loading immediates, and should not be
/// used as a scratch register (this is the `IMM_REG` constant).  `s4-7`/`v4-7`
/// are all available, and are callee-saved.
///
/// For general-purpose registers, `x9-15` (also called `w9-15`) are reasonable
/// choices; they are caller-saved, so we can trash them at will.
pub trait AssemblerT {
    /// Data type used during evaluation.
    ///
    /// This should be a `repr(C)` type, so it can be passed around directly.
    type Data;

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

/// Trait defining SIMD width
pub trait SimdAssembler {
    const SIMD_SIZE: usize;
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

    #[cfg(target_arch = "aarch64")]
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

    #[cfg(target_arch = "x86_64")]
    fn prepare_stack(&mut self, slot_count: usize) {
        unimplemented!()
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

    fn len(&self) -> usize {
        self.len
    }
}

impl From<Mmap> for MmapAssembler {
    fn from(mmap: Mmap) -> Self {
        Self { mmap, len: 0 }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

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
            Op::Reg(UnaryOpcode::Neg, out, arg) => {
                asm.build_neg(out, arg);
            }
            Op::Reg(UnaryOpcode::Abs, out, arg) => {
                asm.build_abs(out, arg);
            }
            Op::Reg(UnaryOpcode::Recip, out, arg) => {
                asm.build_recip(out, arg);
            }
            Op::Reg(UnaryOpcode::Sqrt, out, arg) => {
                asm.build_sqrt(out, arg);
            }
            Op::Reg(UnaryOpcode::Copy, out, arg) => {
                asm.build_copy(out, arg);
            }
            Op::Reg(UnaryOpcode::Square, out, arg) => {
                asm.build_square(out, arg);
            }
            Op::RegReg(BinaryOpcode::Add, out, lhs, rhs) => {
                asm.build_add(out, lhs, rhs);
            }
            Op::RegReg(BinaryOpcode::Mul, out, lhs, rhs) => {
                asm.build_mul(out, lhs, rhs);
            }
            Op::RegReg(BinaryOpcode::Div, out, lhs, rhs) => {
                asm.build_div(out, lhs, rhs);
            }
            Op::RegReg(BinaryOpcode::Sub, out, lhs, rhs) => {
                asm.build_sub(out, lhs, rhs);
            }
            Op::RegReg(BinaryOpcode::Min, out, lhs, rhs) => {
                asm.build_min(out, lhs, rhs);
            }
            Op::RegReg(BinaryOpcode::Max, out, lhs, rhs) => {
                asm.build_max(out, lhs, rhs);
            }
            Op::RegImm(BinaryOpcode::Add, out, arg, imm)
            | Op::ImmReg(BinaryOpcode::Add, out, arg, imm) => {
                asm.build_add_imm(out, arg, imm);
            }
            Op::RegImm(BinaryOpcode::Mul, out, arg, imm)
            | Op::ImmReg(BinaryOpcode::Mul, out, arg, imm) => {
                asm.build_mul_imm(out, arg, imm);
            }
            Op::RegImm(BinaryOpcode::Div, out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, arg, reg);
            }
            Op::ImmReg(BinaryOpcode::Div, out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, reg, arg);
            }
            Op::ImmReg(BinaryOpcode::Sub, out, arg, imm) => {
                asm.build_sub_imm_reg(out, arg, imm);
            }
            Op::RegImm(BinaryOpcode::Sub, out, arg, imm) => {
                asm.build_sub_reg_imm(out, arg, imm);
            }
            Op::RegImm(BinaryOpcode::Min, out, arg, imm)
            | Op::ImmReg(BinaryOpcode::Min, out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_min(out, arg, reg);
            }
            Op::RegImm(BinaryOpcode::Max, out, arg, imm)
            | Op::ImmReg(BinaryOpcode::Max, out, arg, imm) => {
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
    type PointEval = point::JitPointEval;
    type FloatSliceEval = float_slice::JitFloatSliceEval;
    type GradSliceEval = grad_slice::JitGradSliceEval;

    fn tile_sizes_3d() -> &'static [usize] {
        &[64, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[128, 16]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle owning a JIT-compiled tracing function of some kind
///
/// Users are unlikely to use this directly; consider using the
/// [`jit::Eval`](Eval) evaluator family instead.
pub struct JitTracingEval<I: AssemblerT> {
    mmap: Arc<Mmap>,
    var_count: usize,
    fn_trace: unsafe extern "C" fn(
        I::Data,    // X
        I::Data,    // Y
        I::Data,    // Z
        *const f32, // vars
        *mut u8,    // choices
        *mut u8,    // simplify (single boolean)
    ) -> I::Data,
}

impl<I: AssemblerT> Clone for JitTracingEval<I> {
    fn clone(&self) -> Self {
        Self {
            mmap: self.mmap.clone(),
            var_count: self.var_count,
            fn_trace: self.fn_trace,
        }
    }
}

unsafe impl<I: AssemblerT> Send for JitTracingEval<I> {}

impl<I: AssemblerT> EvaluatorStorage<Eval> for JitTracingEval<I> {
    type Storage = Mmap;
    fn new_with_storage(t: &Tape<Eval>, prev: Self::Storage) -> Self {
        let mmap = build_asm_fn_with_storage::<I>(t, prev);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_trace: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.mmap).ok()
    }
}

impl<I: AssemblerT> TracingEvaluator<I::Data, Eval> for JitTracingEval<I> {
    type Data = ();

    /// Evaluates a single point, capturing execution in `choices`
    fn eval_with(
        &self,
        x: I::Data,
        y: I::Data,
        z: I::Data,
        vars: &[f32],
        choices: &mut [Choice],
        _data: &mut (),
    ) -> (I::Data, bool) {
        let mut simplify = 0;
        assert_eq!(vars.len(), self.var_count);
        let out = unsafe {
            (self.fn_trace)(
                x,
                y,
                z,
                vars.as_ptr(),
                choices.as_mut_ptr() as *mut u8,
                &mut simplify,
            )
        };
        (out, simplify != 0)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle owning a JIT-compiled bulk function of some kind
///
/// Users are unlikely to use this directly; consider using the
/// [`jit::Eval`](Eval) evaluator family instead.
pub struct JitBulkEval<I: AssemblerT> {
    mmap: Arc<Mmap>,
    var_count: usize,
    fn_bulk: unsafe extern "C" fn(
        *const f32,   // X
        *const f32,   // Y
        *const f32,   // Z
        *const f32,   // vars
        *mut I::Data, // out
        u64,          // size
    ) -> I::Data,
}

impl<I: AssemblerT> Clone for JitBulkEval<I> {
    fn clone(&self) -> Self {
        Self {
            mmap: self.mmap.clone(),
            var_count: self.var_count,
            fn_bulk: self.fn_bulk,
        }
    }
}

unsafe impl<I: AssemblerT> Send for JitBulkEval<I> {}

impl<I: AssemblerT> EvaluatorStorage<Eval> for JitBulkEval<I> {
    type Storage = Mmap;
    fn new_with_storage(t: &Tape<Eval>, prev: Self::Storage) -> Self {
        let mmap = build_asm_fn_with_storage::<I>(t, prev);
        let ptr = mmap.as_ptr();
        Self {
            mmap: Arc::new(mmap),
            var_count: t.var_count(),
            fn_bulk: unsafe { std::mem::transmute(ptr) },
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.mmap).ok()
    }
}

impl<I: AssemblerT + SimdAssembler> BulkEvaluator<I::Data, Eval>
    for JitBulkEval<I>
where
    I::Data: Copy + From<f32>,
{
    type Data = ();

    /// Evaluate multiple points
    fn eval_with(
        &self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [I::Data],
        _data: &mut (),
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.var_count);

        let n = xs.len();

        // Special case for when we have fewer items than the native SIMD size,
        // in which case the input slices can't be used as workspace (because
        // they are not valid for the entire range of values read in assembly)
        if n < I::SIMD_SIZE {
            // We can't use I::SIMD_SIZE directly here due to Rust limitations,
            // so instead we hard-code it to 4 with an assertion that
            // (hopefully) will be compiled out.
            let mut x = [0.0; 4];
            let mut y = [0.0; 4];
            let mut z = [0.0; 4];
            assert!(I::SIMD_SIZE <= 4);

            x[0..n].copy_from_slice(xs);
            y[0..n].copy_from_slice(ys);
            z[0..n].copy_from_slice(zs);

            let mut tmp = [std::f32::NAN.into(); 4];

            unsafe {
                (self.fn_bulk)(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vars.as_ptr(),
                    tmp.as_mut_ptr(),
                    I::SIMD_SIZE as u64,
                );
            }
            out[0..n].copy_from_slice(&tmp[0..n]);
        } else {
            // Our vectorized function only accepts sets of a particular width,
            // so we'll find the biggest multiple, then do an extra operation to
            // process any remainders.
            let m = (n / I::SIMD_SIZE) * I::SIMD_SIZE; // Round down
            unsafe {
                (self.fn_bulk)(
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    vars.as_ptr(),
                    out.as_mut_ptr(),
                    m as u64,
                );
            }
            // If we weren't given an even multiple of vector width, then we'll
            // handle the remaining items by simply evaluating the *last* full
            // vector in the array again.
            if n != m {
                unsafe {
                    (self.fn_bulk)(
                        xs.as_ptr().add(n - I::SIMD_SIZE),
                        ys.as_ptr().add(n - I::SIMD_SIZE),
                        zs.as_ptr().add(n - I::SIMD_SIZE),
                        vars.as_ptr(),
                        out.as_mut_ptr().add(n - I::SIMD_SIZE),
                        I::SIMD_SIZE as u64,
                    );
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(Eval);
    crate::interval_tests!(Eval);
    crate::float_slice_tests!(Eval);
    crate::point_tests!(Eval);
}
