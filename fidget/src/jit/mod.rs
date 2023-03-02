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
    eval::{
        bulk::BulkEvaluator, tape::Data as TapeData, tracing::TracingEvaluator,
        Choice, EvaluatorStorage, Family, Tape,
    },
    jit::mmap::Mmap,
    vm::Op,
    Error,
};
use dynasmrt::{
    components::PatchLoc, dynasm, AssemblyOffset, DynamicLabel, DynasmApi,
    DynasmError, DynasmLabelApi, TargetKind,
};
use std::sync::Arc;

mod mmap;

// Evaluators
mod float_slice;
mod grad_slice;
mod interval;
mod point;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!(
    "The `jit` module only builds on Linux and macOS; \
    please disable the `jit` feature"
);

#[cfg(target_arch = "aarch64")]
mod arch {
    /// We can use registers v8-v15 (callee saved) and v16-v31 (caller saved)
    pub const REGISTER_LIMIT: u8 = 24;
    pub const OFFSET: u8 = 8;
    pub const IMM_REG: u8 = 3;
}

#[cfg(target_arch = "x86_64")]
mod arch {
    // We use xmm0 for immediates and xmm1-3 for temporaries
    pub const REGISTER_LIMIT: u8 = 12;
    pub const OFFSET: u8 = 4;
    pub const IMM_REG: u8 = 0;
}

/// Number of registers available when executing natively
const REGISTER_LIMIT: u8 = arch::REGISTER_LIMIT;

/// Offset before the first useable register
const OFFSET: u8 = arch::OFFSET;

/// Register written to by `CopyImm`
///
/// `IMM_REG` is selected to avoid scratch registers used by other
/// functions, e.g. interval mul / min / max
const IMM_REG: u8 = arch::IMM_REG;

#[cfg(target_arch = "aarch64")]
type RegIndex = u32;

#[cfg(target_arch = "x86_64")]
type RegIndex = u8;

/// Converts from a tape-local register to an AArch64 register
///
/// Tape-local registers are in the range `0..REGISTER_LIMIT`, while ARM
/// registers have an offset (based on calling convention).
///
/// This uses `wrapping_add` to support immediates, which are loaded into an ARM
/// register below `OFFSET` (which is "negative" from the perspective of this
/// function).
fn reg(r: u8) -> RegIndex {
    let out = r.wrapping_add(OFFSET) as RegIndex;
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

    /// Loads an immediate into a register, returning that register
    fn load_imm(&mut self, imm: f32) -> u8;

    fn finalize(self, out_reg: u8) -> Result<Mmap, Error>;
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
        // We always use the stack on x86_64, if only to store X/Y/Z
        let stack_slots = slot_count.saturating_sub(REGISTER_LIMIT as usize);

        // We put X/Y/Z values at the top of the stack, where they can be
        // accessed with `movss [rbp - i*size_of(T)] xmm`.  This frees up the
        // incoming registers (xmm0-2) in the point evaluator.
        let mem = (stack_slots + 4) * std::mem::size_of::<T>();

        // Round up to the nearest multiple of 16 bytes, for alignment
        self.mem_offset = ((mem + 15) / 16) * 16;
        dynasm!(self.ops
            ; sub rsp, self.mem_offset as i32
        );
    }

    fn stack_pos(&self, slot: u32) -> u32 {
        assert!(slot >= REGISTER_LIMIT as u32);
        (slot - REGISTER_LIMIT as u32) * std::mem::size_of::<T>() as u32
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(target_arch = "x86_64")]
type Relocation = dynasmrt::x64::X64Relocation;

#[cfg(target_arch = "aarch64")]
type Relocation = dynasmrt::aarch64::Aarch64Relocation;

struct MmapAssembler {
    mmap: Mmap,
    len: usize,

    global_labels: [Option<AssemblyOffset>; 26],
    local_labels: [Option<AssemblyOffset>; 26],

    global_relocs: arrayvec::ArrayVec<(PatchLoc<Relocation>, u8), 1>,
    local_relocs: arrayvec::ArrayVec<(PatchLoc<Relocation>, u8), 8>,
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

// This is a very limited implementation of the labels API.  Compared to the
// standard labels API, it has the following limitations:
//
// - Labels must be a single character
// - Local labels must be committed before they're reused, using `commit_local`
// - Only 8 local jumps are available at any given time; this is reset when
//   `commit_local` is called.  (if this becomes problematic, it can be
//   increased by tweaking the size of `local_relocs: ArrayVec<..., 8>`.
//
// In exchange for these limitations, it allocates no memory at runtime, and all
// label lookups are done in constant time.
impl DynasmLabelApi for MmapAssembler {
    type Relocation = Relocation;

    fn local_label(&mut self, name: &'static str) {
        if name.len() != 1 {
            panic!("local label must be a single character");
        }
        let c = name.as_bytes()[0].wrapping_sub(b'A');
        if c >= 26 {
            panic!("Invalid label {name}, must be A-Z");
        }
        if self.local_labels[c as usize].is_some() {
            panic!("duplicate local label {name}");
        }

        self.local_labels[c as usize] = Some(self.offset());
    }
    fn global_label(&mut self, name: &'static str) {
        if name.len() != 1 {
            panic!("local label must be a single character");
        }
        let c = name.as_bytes()[0].wrapping_sub(b'A');
        if c >= 26 {
            panic!("Invalid label {name}, must be A-Z");
        }
        if self.global_labels[c as usize].is_some() {
            panic!("duplicate global label {name}");
        }

        self.global_labels[c as usize] = Some(self.offset());
    }
    fn dynamic_label(&mut self, _id: DynamicLabel) {
        panic!("dynamic labels are not supported");
    }
    fn global_relocation(
        &mut self,
        name: &'static str,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: Relocation,
    ) {
        let location = self.offset();
        if name.len() != 1 {
            panic!("local label must be a single character");
        }
        let c = name.as_bytes()[0].wrapping_sub(b'A');
        if c >= 26 {
            panic!("Invalid label {name}, must be A-Z");
        }
        self.global_relocs.push((
            PatchLoc::new(
                location,
                target_offset,
                field_offset,
                ref_offset,
                kind,
            ),
            c,
        ));
    }
    fn dynamic_relocation(
        &mut self,
        _id: DynamicLabel,
        _target_offset: isize,
        _field_offset: u8,
        _ref_offset: u8,
        _kind: Relocation,
    ) {
        panic!("dynamic relocations are not supported");
    }
    fn forward_relocation(
        &mut self,
        name: &'static str,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: Relocation,
    ) {
        if name.len() != 1 {
            panic!("local label must be a single character");
        }
        let c = name.as_bytes()[0].wrapping_sub(b'A');
        if c >= 26 {
            panic!("Invalid label {name}, must be A-Z");
        }
        if self.local_labels[c as usize].is_some() {
            panic!("invalid forward relocation: {name} already exists!");
        }
        let location = self.offset();
        self.local_relocs.push((
            PatchLoc::new(
                location,
                target_offset,
                field_offset,
                ref_offset,
                kind,
            ),
            c,
        ));
    }
    fn backward_relocation(
        &mut self,
        name: &'static str,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: Relocation,
    ) {
        if name.len() != 1 {
            panic!("local label must be a single character");
        }
        let c = name.as_bytes()[0].wrapping_sub(b'A');
        if c >= 26 {
            panic!("Invalid label {name}, must be A-Z");
        }
        if self.local_labels[c as usize].is_none() {
            panic!("invalid backward relocation: {name} does not exist");
        }
        let location = self.offset();
        self.local_relocs.push((
            PatchLoc::new(
                location,
                target_offset,
                field_offset,
                ref_offset,
                kind,
            ),
            c,
        ));
    }
    fn bare_relocation(
        &mut self,
        _target: usize,
        _field_offset: u8,
        _ref_offset: u8,
        _kind: Relocation,
    ) {
        panic!("bare relocations not implemented");
    }
}

impl MmapAssembler {
    fn commit_local(&mut self) -> Result<(), Error> {
        let baseaddr = self.mmap.as_ptr() as usize;

        for (loc, label) in self.local_relocs.take() {
            let target =
                self.local_labels.get(label as usize).unwrap().unwrap();
            let buf = &mut self.mmap.as_mut_slice()[loc.range(0)];
            if loc.patch(buf, baseaddr, target.0).is_err() {
                return Err(DynasmError::ImpossibleRelocation(
                    TargetKind::Local("oh no"),
                )
                .into());
            }
        }
        self.local_labels = [None; 26];
        Ok(())
    }

    fn finalize(mut self) -> Result<Mmap, Error> {
        self.commit_local()?;

        let baseaddr = self.mmap.as_ptr() as usize;
        for (loc, label) in self.global_relocs.take() {
            let target =
                self.global_labels.get(label as usize).unwrap().unwrap();
            let buf = &mut self.mmap.as_mut_slice()[loc.range(0)];
            if loc.patch(buf, baseaddr, target.0).is_err() {
                return Err(DynasmError::ImpossibleRelocation(
                    TargetKind::Global("oh no"),
                )
                .into());
            }
        }

        self.mmap.finalize(self.len);
        Ok(self.mmap)
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
        Self {
            mmap,
            len: 0,
            global_labels: [None; 26],
            local_labels: [None; 26],
            global_relocs: Default::default(),
            local_relocs: Default::default(),
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

fn build_asm_fn_with_storage<A: AssemblerT>(t: &TapeData, s: Mmap) -> Mmap {
    // This guard may be a unit value on some systems
    #[allow(clippy::let_unit_value)]
    let _guard = Mmap::thread_mode_write();

    s.make_write();
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

    asm.finalize(0).expect("failed to build JIT function")
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
    // TODO: make this `sysv64` on `x86_64` machines
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
            // We can't use I::SIMD_SIZE directly here due to Rust limitations.
            // Instead we hard-code a maximum SIMD size along with an assertion
            // that should be optimized out; we can't use a constant assertion
            // here due to the same compiler limitations.
            const MAX_SIMD_WIDTH: usize = 8;
            let mut x = [0.0; MAX_SIMD_WIDTH];
            let mut y = [0.0; MAX_SIMD_WIDTH];
            let mut z = [0.0; MAX_SIMD_WIDTH];
            assert!(I::SIMD_SIZE <= MAX_SIMD_WIDTH);

            x[0..n].copy_from_slice(xs);
            y[0..n].copy_from_slice(ys);
            z[0..n].copy_from_slice(zs);

            let mut tmp = [std::f32::NAN.into(); MAX_SIMD_WIDTH];

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
