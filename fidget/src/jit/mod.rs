//! Compilation down to native machine code
//!
//! Users are unlikely to use anything in this module other than [`Eval`], which
//! is a [`Family`] of JIT evaluators.
//!
//! ```
//! use fidget::{rhai, eval::{TracingEvaluator, Shape}, jit::JitShape};
//!
//! let (sum, ctx) = rhai::eval("x + y")?;
//! let shape = JitShape::new(&ctx, sum)?;
//!
//! // Generate machine code to execute the tape
//! let tape = shape.point_tape();
//! let mut eval = JitShape::new_point_eval();
//!
//! // This calls directly into that machine code!
//! let (r, _trace) = eval.eval(&tape, 0.1, 0.3, 0.0, &[])?;
//! assert_eq!(r, 0.1 + 0.3);
//! # Ok::<(), fidget::Error>(())
//! ```

use crate::{
    compiler::RegOp,
    context::{Context, Node},
    eval::{
        bulk::BulkEvaluator,
        tracing::TracingEvaluator,
        types::{Grad, Interval},
        Choice, Shape, ShapeVars, Tape, TapeData,
    },
    jit::mmap::Mmap,
    vm::GenericVmShape,
    Error,
};
use dynasmrt::{
    components::PatchLoc, dynasm, AssemblyOffset, DynamicLabel, DynasmApi,
    DynasmError, DynasmLabelApi, TargetKind,
};
use std::{collections::HashMap, sync::Arc};

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
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as arch;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as arch;

/// Number of registers available when executing natively
const REGISTER_LIMIT: u8 = arch::REGISTER_LIMIT;

/// Offset before the first useable register
const OFFSET: u8 = arch::OFFSET;

/// Register written to by `CopyImm`
///
/// It is the responsibility of functions to avoid writing to `IMM_REG` in cases
/// where it could be one of their arguments (i.e. all functions of 2 or more
/// arguments).
const IMM_REG: u8 = arch::IMM_REG;

/// Type for a register index in `dynasm` code
#[cfg(target_arch = "aarch64")]
type RegIndex = u32;

/// Type for a register index in `dynasm` code
#[cfg(target_arch = "x86_64")]
type RegIndex = u8;

/// Converts from a tape-local register to a hardware register
///
/// Tape-local registers are in the range `0..REGISTER_LIMIT`, while ARM
/// registers have an offset (based on calling convention).
///
/// This uses `wrapping_add` to support immediates, which are loaded into a
/// register below [`OFFSET`] (which is "negative" from the perspective of this
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
pub trait AssemblerT {
    /// Data type used during evaluation.
    ///
    /// This should be a `repr(C)` type, so it can be passed around directly.
    type Data;

    /// Initializes the assembler with the given slot count
    ///
    /// This will likely construct a function prelude and reserve space on the
    /// stack for slot spills.
    fn init(m: Mmap, slot_count: usize) -> Self;

    /// Builds a load from memory to a register
    fn build_load(&mut self, dst_reg: u8, src_mem: u32);

    /// Builds a store from a register to a memory location
    fn build_store(&mut self, dst_mem: u32, src_reg: u8);

    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u8);

    /// Copies a variable (provided in an input array) to `out_reg`
    fn build_var(&mut self, out_reg: u8, src_arg: u32);

    /// Copies a register
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8);

    /// Unary negation
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8);

    /// Absolute value
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8);

    /// Reciprocal (1 / `lhs_reg`)
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8);

    /// Square root
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8);

    /// Square
    ///
    /// This has a default implementation, but can be overloaded for efficiency;
    /// for example, in interval arithmetic, we benefit from knowing that both
    /// values are the same.
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        self.build_mul(out_reg, lhs_reg, lhs_reg)
    }

    /// Addition
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Subtraction
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Multiplication
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Division
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Maximum of two values
    ///
    /// In a tracing evaluator, this function must also write to the `choices`
    /// array and may set `simplify` if one branch is always taken.
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Minimum of two values
    ///
    /// In a tracing evaluator, this function must also write to the `choices`
    /// array and may set `simplify` if one branch is always taken.
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    // Special-case functions for immediates.  In some cases, you can be more
    // efficient if you know that an argument is an immediate (for example, both
    // values in the interval will be the same, and it wlll have no gradients).

    /// Builds a addition (immediate + register)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_add_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_add(out_reg, lhs_reg, imm);
    }
    /// Builds a subtraction (immediate − register)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_sub_imm_reg(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_sub(out_reg, imm, arg);
    }
    /// Builds a subtraction (register − immediate)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_sub_reg_imm(&mut self, out_reg: u8, arg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_sub(out_reg, arg, imm);
    }
    /// Builds a multiplication (register × immediate)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_mul_imm(&mut self, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = self.load_imm(imm);
        self.build_mul(out_reg, lhs_reg, imm);
    }

    /// Loads an immediate into a register, returning that register
    fn load_imm(&mut self, imm: f32) -> u8;

    /// Finalize the assembly code, returning a memory-mapped region
    fn finalize(self, out_reg: u8) -> Result<Mmap, Error>;
}

/// Trait defining SIMD width
pub trait SimdSize {
    /// Number of elements processed in a single iteration
    ///
    /// This value is used when checking array sizes, as we want to be sure to
    /// pass the JIT code an appropriately sized array.
    const SIMD_SIZE: usize;
}

/////////////////////////////////////////////////////////////////////////////////////////

pub(crate) struct AssemblerData<T> {
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

/// This is a very limited implementation of the labels API.  Compared to the
/// standard labels API, it has the following limitations:
///
/// - Labels must be a single character
/// - Local labels must be committed before they're reused, using `commit_local`
/// - Only 8 local jumps are available at any given time; this is reset when
///   `commit_local` is called.  (if this becomes problematic, it can be
///   increased by tweaking the size of `local_relocs: ArrayVec<..., 8>`.
///
/// In exchange for these limitations, it allocates no memory at runtime, and all
/// label lookups are done in constant time.
///
/// However, it still has overhead compared to computing the jumps by hand;
/// this overhead was roughly 5% in one unscientific test.
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
    /// Applies all local relocations, clearing the `local_relocs` array
    ///
    /// This should be called after any function which uses local labels.
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

fn build_asm_fn_with_storage<A: AssemblerT>(
    t: &TapeData<REGISTER_LIMIT>,
    s: Mmap,
) -> Mmap {
    // This guard may be a unit value on some systems
    #[allow(clippy::let_unit_value)]
    let _guard = Mmap::thread_mode_write();

    s.make_write();
    let mut asm = A::init(s, t.slot_count());

    for op in t.iter_asm() {
        match op {
            RegOp::Load(reg, mem) => {
                asm.build_load(reg, mem);
            }
            RegOp::Store(reg, mem) => {
                asm.build_store(mem, reg);
            }
            RegOp::Input(out, i) => {
                asm.build_input(out, i);
            }
            RegOp::Var(out, i) => {
                asm.build_var(out, i);
            }
            RegOp::NegReg(out, arg) => {
                asm.build_neg(out, arg);
            }
            RegOp::AbsReg(out, arg) => {
                asm.build_abs(out, arg);
            }
            RegOp::RecipReg(out, arg) => {
                asm.build_recip(out, arg);
            }
            RegOp::SqrtReg(out, arg) => {
                asm.build_sqrt(out, arg);
            }
            RegOp::CopyReg(out, arg) => {
                asm.build_copy(out, arg);
            }
            RegOp::SquareReg(out, arg) => {
                asm.build_square(out, arg);
            }
            RegOp::AddRegReg(out, lhs, rhs) => {
                asm.build_add(out, lhs, rhs);
            }
            RegOp::MulRegReg(out, lhs, rhs) => {
                asm.build_mul(out, lhs, rhs);
            }
            RegOp::DivRegReg(out, lhs, rhs) => {
                asm.build_div(out, lhs, rhs);
            }
            RegOp::SubRegReg(out, lhs, rhs) => {
                asm.build_sub(out, lhs, rhs);
            }
            RegOp::MinRegReg(out, lhs, rhs) => {
                asm.build_min(out, lhs, rhs);
            }
            RegOp::MaxRegReg(out, lhs, rhs) => {
                asm.build_max(out, lhs, rhs);
            }
            RegOp::AddRegImm(out, arg, imm) => {
                asm.build_add_imm(out, arg, imm);
            }
            RegOp::MulRegImm(out, arg, imm) => {
                asm.build_mul_imm(out, arg, imm);
            }
            RegOp::DivRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, arg, reg);
            }
            RegOp::DivImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, reg, arg);
            }
            RegOp::SubImmReg(out, arg, imm) => {
                asm.build_sub_imm_reg(out, arg, imm);
            }
            RegOp::SubRegImm(out, arg, imm) => {
                asm.build_sub_reg_imm(out, arg, imm);
            }
            RegOp::MinRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_min(out, arg, reg);
            }
            RegOp::MaxRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_max(out, arg, reg);
            }
            RegOp::CopyImm(out, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_copy(out, reg);
            }
        }
    }

    asm.finalize(0).expect("failed to build JIT function")
    // JIT execute mode is restored here when the _guard is dropped
}

/// Shape for use with a JIT evaluator
#[derive(Clone)]
pub struct JitShape(GenericVmShape<REGISTER_LIMIT>);

impl JitShape {
    /// Build a new shape for the given node
    pub fn new(ctx: &Context, node: Node) -> Result<Self, Error> {
        GenericVmShape::new(ctx, node).map(JitShape)
    }
    fn tracing_tape<A: AssemblerT>(
        &self,
        storage: Option<Mmap>,
    ) -> JitTracingFn<A::Data> {
        let storage = storage.unwrap_or_else(|| Mmap::new(0).unwrap());
        let f = build_asm_fn_with_storage::<A>(self.0.data(), storage);
        let ptr = f.as_ptr();
        JitTracingFn {
            mmap: f,
            var_count: self.0.var_count(),
            choice_count: self.0.choice_count(),
            fn_trace: unsafe { std::mem::transmute(ptr) },
        }
    }
    fn bulk_tape<A: AssemblerT>(
        &self,
        storage: Option<Mmap>,
    ) -> JitBulkFn<A::Data> {
        let storage = storage.unwrap_or_else(|| Mmap::new(0).unwrap());
        let f = build_asm_fn_with_storage::<A>(self.0.data(), storage);
        let ptr = f.as_ptr();
        JitBulkFn {
            mmap: f,
            var_count: self.0.var_count(),
            fn_bulk: unsafe { std::mem::transmute(ptr) },
        }
    }
}

impl Shape for JitShape {
    type Trace = Vec<Choice>;
    type Storage = TapeData<REGISTER_LIMIT>;
    type TapeStorage = Mmap;

    type IntervalEval = JitTracingEval;
    type PointEval = JitTracingEval;
    type FloatSliceEval = JitBulkEval<f32>;
    type GradSliceEval = JitBulkEval<Grad>;

    fn point_tape(&self, storage: Option<Mmap>) -> JitTracingFn<f32> {
        self.tracing_tape::<point::PointAssembler>(storage)
    }

    fn interval_tape(&self, storage: Option<Mmap>) -> JitTracingFn<Interval> {
        self.tracing_tape::<interval::IntervalAssembler>(storage)
    }

    fn float_slice_tape(&self, storage: Option<Mmap>) -> JitBulkFn<f32> {
        self.bulk_tape::<float_slice::FloatSliceAssembler>(storage)
    }

    fn grad_slice_tape(&self, storage: Option<Mmap>) -> JitBulkFn<Grad> {
        self.bulk_tape::<grad_slice::GradSliceAssembler>(storage)
    }

    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Option<Self::Storage>,
    ) -> Result<Self, Error> {
        self.0.simplify_inner(trace, storage).map(JitShape)
    }

    fn recycle(self) -> Option<Self::Storage> {
        self.0.recycle()
    }

    fn tile_sizes_3d() -> &'static [usize] {
        &[64, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[128, 16]
    }

    fn simplify_tree_during_meshing(d: usize) -> bool {
        // Unscientifically selected, but similar to tile_sizes_3d
        d % 8 == 4
    }

    #[cfg(test)]
    fn size(&self) -> usize {
        self.0.size()
    }
}

////////////////////////////////////////////////////////////////////////////////

// Selects the calling convention based on platform; this is forward-looking for
// eventual x86 Windows support, where we still want to use the sysv64 calling
// convention.
/// Macro to build a function type with a `extern "sysv64"` calling convention
///
/// This is selected at compile time, based on `target_arch`
#[cfg(target_arch = "x86_64")]
macro_rules! jit_fn {
    (unsafe fn($($args:tt)*) -> $($out:tt)*) => {
        unsafe extern "sysv64" fn($($args)*) -> $($out)*
    };
}

/// Macro to build a function type with the `extern "C"` calling convention
///
/// This is selected at compile time, based on `target_arch`
#[cfg(target_arch = "aarch64")]
macro_rules! jit_fn {
    (unsafe fn($($args:tt)*) -> $($out:tt)*) => {
        unsafe extern "C" fn($($args)*) -> $($out)*
    };
}

////////////////////////////////////////////////////////////////////////////////

/// Handle owning a JIT-compiled tracing function of some kind
///
/// Users are unlikely to use this directly; consider using the
/// [`jit::Eval`](Eval) evaluator family instead.
#[derive(Default)]
pub struct JitTracingEval {
    choices: Vec<Choice>,
}

/// Handle to an owned function pointer for tracing evaluation
pub struct JitTracingFn<T> {
    #[allow(unused)]
    mmap: Mmap,
    var_count: usize,
    choice_count: usize,
    fn_trace: jit_fn!(
        unsafe fn(
            T,          // X
            T,          // Y
            T,          // Z
            *const f32, // vars
            *mut u8,    // choices
            *mut u8,    // simplify (single boolean)
        ) -> T
    ),
}

impl<T> Tape for JitTracingFn<T> {
    type Storage = Mmap;
    fn recycle(self) -> Self::Storage {
        self.mmap
    }
}

// SAFETY: there is no mutable state in a `JitTracingFn`, and the pointer
// inside of it points to its own `Mmap`, which is owned by an `Arc`
unsafe impl<T> Send for JitTracingFn<T> {}
unsafe impl<T> Sync for JitTracingFn<T> {}

impl<T: From<f32>> TracingEvaluator<T> for JitTracingEval {
    type Tape = JitTracingFn<T>;
    type Trace = Vec<Choice>;
    type TapeStorage = Mmap;

    /// Evaluates a single point, capturing an evaluation trace
    fn eval<F: Into<T>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(T, Option<&Vec<Choice>>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let mut simplify = 0;
        self.choices.resize(tape.choice_count, Choice::Unknown);
        self.choices.fill(Choice::Unknown);
        if vars.len() != tape.var_count {
            return Err(Error::BadVarSlice(vars.len(), tape.var_count));
        }
        let out = unsafe {
            (tape.fn_trace)(
                x,
                y,
                z,
                vars.as_ptr(),
                self.choices.as_mut_ptr() as *mut u8,
                &mut simplify,
            )
        };
        Ok((
            out,
            if simplify != 0 {
                Some(&self.choices)
            } else {
                None
            },
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle to an owned function pointer for bulk evaluation
pub struct JitBulkFn<T> {
    #[allow(unused)]
    mmap: Mmap,
    var_count: usize,
    fn_bulk: jit_fn!(
        unsafe fn(
            *const f32, // X
            *const f32, // Y
            *const f32, // Z
            *const f32, // vars
            *mut T,     // out
            u64,        // size
        ) -> T
    ),
}

impl<T> Tape for JitBulkFn<T> {
    type Storage = Mmap;
    fn recycle(self) -> Self::Storage {
        self.mmap
    }
}

/// Bulk evaluator for JIT functions
pub struct JitBulkEval<T> {
    /// Output array that's written to during evaluation
    out: Vec<T>,
}

impl<T> Default for JitBulkEval<T> {
    fn default() -> Self {
        Self { out: vec![] }
    }
}

// SAFETY: there is no mutable state in a `JitBulkFn`, and the pointer
// inside of it points to its own `Mmap`, which is owned by an `Arc`
unsafe impl<T> Send for JitBulkFn<T> {}
unsafe impl<T> Sync for JitBulkFn<T> {}

impl<T> BulkEvaluator<T> for JitBulkEval<T>
where
    T: From<f32> + Copy + SimdSize,
{
    type Tape = JitBulkFn<T>;
    type TapeStorage = Mmap;

    /// Evaluate multiple points
    fn eval(
        &mut self,
        tape: &Self::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<&[T], Error> {
        self.check_arguments(xs, ys, zs, vars, tape.var_count)?;

        let n = xs.len();
        self.out.resize(n, f32::NAN.into());
        self.out.fill(f32::NAN.into());

        // Special case for when we have fewer items than the native SIMD size,
        // in which case the input slices can't be used as workspace (because
        // they are not valid for the entire range of values read in assembly)
        if n < T::SIMD_SIZE {
            // We can't use T::SIMD_SIZE directly here due to Rust limitations.
            // Instead we hard-code a maximum SIMD size along with an assertion
            // that should be optimized out; we can't use a constant assertion
            // here due to the same compiler limitations.
            const MAX_SIMD_WIDTH: usize = 8;
            let mut x = [0.0; MAX_SIMD_WIDTH];
            let mut y = [0.0; MAX_SIMD_WIDTH];
            let mut z = [0.0; MAX_SIMD_WIDTH];
            assert!(T::SIMD_SIZE <= MAX_SIMD_WIDTH);

            x[0..n].copy_from_slice(xs);
            y[0..n].copy_from_slice(ys);
            z[0..n].copy_from_slice(zs);

            let mut tmp = [std::f32::NAN.into(); MAX_SIMD_WIDTH];

            unsafe {
                (tape.fn_bulk)(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vars.as_ptr(),
                    tmp.as_mut_ptr(),
                    T::SIMD_SIZE as u64,
                );
            }
            self.out.copy_from_slice(&tmp[0..n]);
        } else {
            // Our vectorized function only accepts sets of a particular width,
            // so we'll find the biggest multiple, then do an extra operation to
            // process any remainders.
            let m = (n / T::SIMD_SIZE) * T::SIMD_SIZE; // Round down
            unsafe {
                (tape.fn_bulk)(
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    vars.as_ptr(),
                    self.out.as_mut_ptr(),
                    m as u64,
                );
            }
            // If we weren't given an even multiple of vector width, then we'll
            // handle the remaining items by simply evaluating the *last* full
            // vector in the array again.
            if n != m {
                unsafe {
                    (tape.fn_bulk)(
                        xs.as_ptr().add(n - T::SIMD_SIZE),
                        ys.as_ptr().add(n - T::SIMD_SIZE),
                        zs.as_ptr().add(n - T::SIMD_SIZE),
                        vars.as_ptr(),
                        self.out.as_mut_ptr().add(n - T::SIMD_SIZE),
                        T::SIMD_SIZE as u64,
                    );
                }
            }
        }
        Ok(&self.out)
    }
}

#[cfg(test)]
impl TryFrom<(&Context, Node)> for JitShape {
    type Error = Error;
    fn try_from(c: (&Context, Node)) -> Result<Self, Error> {
        JitShape::new(c.0, c.1)
    }
}

impl ShapeVars for JitShape {
    fn vars(&self) -> Arc<HashMap<String, u32>> {
        self.0.vars()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(JitShape);
    crate::interval_tests!(JitShape);
    crate::float_slice_tests!(JitShape);
    crate::point_tests!(JitShape);
}
