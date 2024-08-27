//! Compilation down to native machine code
//!
//! Users are unlikely to use anything in this module other than [`JitFunction`],
//! which is a [`Function`] that uses JIT evaluation.
//!
//! ```
//! use fidget::{
//!     context::Tree,
//!     shape::EzShape,
//!     jit::JitShape,
//! };
//!
//! let tree = Tree::x() + Tree::y();
//! let shape = JitShape::from(tree);
//!
//! // Generate machine code to execute the tape
//! let tape = shape.ez_point_tape();
//! let mut eval = JitShape::new_point_eval();
//!
//! // This calls directly into that machine code!
//! let (r, _trace) = eval.eval(&tape, 0.1, 0.3, 0.0)?;
//! assert_eq!(r, 0.1 + 0.3);
//! # Ok::<(), fidget::Error>(())
//! ```

use crate::{
    compiler::RegOp,
    context::{Context, Node},
    eval::{
        BulkEvaluator, BulkOutput, Function, MathFunction, Tape,
        TracingEvaluator,
    },
    jit::mmap::Mmap,
    shape::RenderHints,
    types::{Grad, Interval},
    var::VarMap,
    vm::{Choice, GenericVmFunction, VmData, VmTrace, VmWorkspace},
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

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows"
)))]
compile_error!(
    "The `jit` module only builds on Linux, macOS, and Windows; \
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
const REGISTER_LIMIT: usize = arch::REGISTER_LIMIT;

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
trait Assembler {
    /// Data type used during evaluation.
    ///
    /// This should be a `repr(C)` type, so it can be passed around directly.
    type Data;

    /// Initializes the assembler with the given slot count
    ///
    /// This will likely construct a function prelude and reserve space on the
    /// stack for slot spills.
    fn init(m: Mmap, slot_count: usize) -> Self;

    /// Returns an approximate bytes per clause value, used for preallocation
    fn bytes_per_clause() -> usize {
        8 // probably wrong!
    }

    /// Builds a load from memory to a register
    fn build_load(&mut self, dst_reg: u8, src_mem: u32);

    /// Builds a store from a register to a memory location
    fn build_store(&mut self, dst_mem: u32, src_reg: u8);

    /// Copies the given input to `out_reg`
    fn build_input(&mut self, out_reg: u8, src_arg: u32);

    /// Writes the argument register to the output
    fn build_output(&mut self, arg_reg: u8, out_index: u32);

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

    /// Sine
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8);

    /// Cosine
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8);

    /// Tangent
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8);

    /// Arcsine
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8);

    /// Arccosine
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8);

    /// Arctangent
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8);

    /// Exponent
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8);

    /// Natural log
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8);

    /// Less than
    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Square
    ///
    /// This has a default implementation, but can be overloaded for efficiency;
    /// for example, in interval arithmetic, we benefit from knowing that both
    /// values are the same.
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        self.build_mul(out_reg, lhs_reg, lhs_reg)
    }

    /// Arithmetic floor
    fn build_floor(&mut self, out_reg: u8, lhs_reg: u8);

    /// Arithmetic ceiling
    fn build_ceil(&mut self, out_reg: u8, lhs_reg: u8);

    /// Rounding
    fn build_round(&mut self, out_reg: u8, lhs_reg: u8);

    /// Logical not
    fn build_not(&mut self, out_reg: u8, lhs_reg: u8);

    /// Logical and (short-circuiting)
    fn build_and(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Logical or (short-circuiting)
    fn build_or(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Addition
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Subtraction
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Multiplication
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Division
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Four-quadrant arctangent
    fn build_atan2(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

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

    /// Modulo of two values (least non-negative remainder)
    fn build_mod(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    // Special-case functions for immediates.  In some cases, you can be more
    // efficient if you know that an argument is an immediate (for example, both
    // values in the interval will be the same, and it will have no gradients).

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
    fn finalize(self) -> Result<Mmap, Error>;
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

    /// Set to true if we have saved certain callee-saved registers
    ///
    /// These registers are only modified in function calls, so normally we
    /// don't save them.
    saved_callee_regs: bool,

    _p: std::marker::PhantomData<*const T>,
}

impl<T> AssemblerData<T> {
    fn new(mmap: Mmap) -> Self {
        Self {
            ops: MmapAssembler::from(mmap),
            mem_offset: 0,
            saved_callee_regs: false,
            _p: std::marker::PhantomData,
        }
    }

    fn prepare_stack(&mut self, slot_count: usize, stack_size: usize) {
        // We always use the stack, if only to store callee-saved registers
        let mem = slot_count.saturating_sub(REGISTER_LIMIT)
            * std::mem::size_of::<T>()
            + stack_size;

        // Round up to the nearest multiple of 16 bytes, for alignment
        self.mem_offset = ((mem + 15) / 16) * 16;
        self.push_stack();
    }

    #[cfg(target_arch = "aarch64")]
    fn push_stack(&mut self) {
        assert!(self.mem_offset < 4096);
        dynasm!(self.ops
            ; sub sp, sp, self.mem_offset as u32
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn push_stack(&mut self) {
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

    global_relocs: arrayvec::ArrayVec<(PatchLoc<Relocation>, u8), 2>,
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
                self.local_labels[label as usize].expect("invalid local label");
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

fn build_asm_fn_with_storage<A: Assembler>(
    t: &VmData<REGISTER_LIMIT>,
    mut s: Mmap,
) -> Mmap {
    // This guard may be a unit value on some systems
    #[cfg(target_os = "macos")]
    let _guard = Mmap::thread_mode_write();

    let size_estimate = t.len() * A::bytes_per_clause();
    if size_estimate > 2 * s.len() {
        s = Mmap::new(size_estimate).expect("failed to build mmap")
    }

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
            RegOp::Output(arg, i) => {
                asm.build_output(arg, i);
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
            RegOp::SinReg(out, arg) => {
                asm.build_sin(out, arg);
            }
            RegOp::CosReg(out, arg) => {
                asm.build_cos(out, arg);
            }
            RegOp::TanReg(out, arg) => {
                asm.build_tan(out, arg);
            }
            RegOp::AsinReg(out, arg) => {
                asm.build_asin(out, arg);
            }
            RegOp::AcosReg(out, arg) => {
                asm.build_acos(out, arg);
            }
            RegOp::AtanReg(out, arg) => {
                asm.build_atan(out, arg);
            }
            RegOp::ExpReg(out, arg) => {
                asm.build_exp(out, arg);
            }
            RegOp::LnReg(out, arg) => {
                asm.build_ln(out, arg);
            }
            RegOp::CopyReg(out, arg) => {
                asm.build_copy(out, arg);
            }
            RegOp::SquareReg(out, arg) => {
                asm.build_square(out, arg);
            }
            RegOp::FloorReg(out, arg) => {
                asm.build_floor(out, arg);
            }
            RegOp::CeilReg(out, arg) => {
                asm.build_ceil(out, arg);
            }
            RegOp::RoundReg(out, arg) => {
                asm.build_round(out, arg);
            }
            RegOp::NotReg(out, arg) => {
                asm.build_not(out, arg);
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
            RegOp::AtanRegReg(out, lhs, rhs) => {
                asm.build_atan2(out, lhs, rhs);
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
            RegOp::AtanRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_atan2(out, arg, reg);
            }
            RegOp::AtanImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_atan2(out, reg, arg);
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
            RegOp::ModRegReg(out, lhs, rhs) => {
                asm.build_mod(out, lhs, rhs);
            }
            RegOp::ModRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_mod(out, arg, reg);
            }
            RegOp::ModImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_mod(out, reg, arg);
            }
            RegOp::AndRegReg(out, lhs, rhs) => {
                asm.build_and(out, lhs, rhs);
            }
            RegOp::AndRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_and(out, arg, reg);
            }
            RegOp::OrRegReg(out, lhs, rhs) => {
                asm.build_or(out, lhs, rhs);
            }
            RegOp::OrRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_or(out, arg, reg);
            }
            RegOp::CopyImm(out, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_copy(out, reg);
            }
            RegOp::CompareRegReg(out, lhs, rhs) => {
                asm.build_compare(out, lhs, rhs);
            }
            RegOp::CompareRegImm(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_compare(out, arg, reg);
            }
            RegOp::CompareImmReg(out, arg, imm) => {
                let reg = asm.load_imm(imm);
                asm.build_compare(out, reg, arg);
            }
        }
    }

    asm.finalize().expect("failed to build JIT function")
    // JIT execute mode is restored here when the _guard is dropped
}

/// Function for use with a JIT evaluator
#[derive(Clone)]
pub struct JitFunction(GenericVmFunction<REGISTER_LIMIT>);

impl JitFunction {
    fn tracing_tape<A: Assembler>(
        &self,
        storage: Mmap,
    ) -> JitTracingFn<A::Data> {
        let f = build_asm_fn_with_storage::<A>(self.0.data(), storage);
        let ptr = f.as_ptr();
        JitTracingFn {
            mmap: f,
            vars: self.0.data().vars.clone(),
            choice_count: self.0.choice_count(),
            output_count: self.0.output_count(),
            fn_trace: unsafe { std::mem::transmute(ptr) },
        }
    }
    fn bulk_tape<A: Assembler>(&self, storage: Mmap) -> JitBulkFn<A::Data> {
        let f = build_asm_fn_with_storage::<A>(self.0.data(), storage);
        let ptr = f.as_ptr();
        JitBulkFn {
            mmap: f,
            vars: self.0.data().vars.clone(),
            fn_bulk: unsafe { std::mem::transmute(ptr) },
        }
    }
}

impl Function for JitFunction {
    type Trace = VmTrace;
    type Storage = VmData<REGISTER_LIMIT>;
    type Workspace = VmWorkspace<REGISTER_LIMIT>;

    type TapeStorage = Mmap;

    type IntervalEval = JitIntervalEval;
    type PointEval = JitPointEval;
    type FloatSliceEval = JitFloatSliceEval;
    type GradSliceEval = JitGradSliceEval;

    fn point_tape(&self, storage: Mmap) -> JitTracingFn<f32> {
        self.tracing_tape::<point::PointAssembler>(storage)
    }

    fn interval_tape(&self, storage: Mmap) -> JitTracingFn<Interval> {
        self.tracing_tape::<interval::IntervalAssembler>(storage)
    }

    fn float_slice_tape(&self, storage: Mmap) -> JitBulkFn<f32> {
        self.bulk_tape::<float_slice::FloatSliceAssembler>(storage)
    }

    fn grad_slice_tape(&self, storage: Mmap) -> JitBulkFn<Grad> {
        self.bulk_tape::<grad_slice::GradSliceAssembler>(storage)
    }

    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error> {
        self.0
            .simplify_inner(trace.as_slice(), storage, workspace)
            .map(JitFunction)
    }

    fn recycle(self) -> Option<Self::Storage> {
        self.0.recycle()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl RenderHints for JitFunction {
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
    (unsafe fn($($args:tt)*)) => {
        unsafe extern "sysv64" fn($($args)*)
    };
}

/// Macro to build a function type with the `extern "C"` calling convention
///
/// This is selected at compile time, based on `target_arch`
#[cfg(target_arch = "aarch64")]
macro_rules! jit_fn {
    (unsafe fn($($args:tt)*)) => {
        unsafe extern "C" fn($($args)*)
    };
}

////////////////////////////////////////////////////////////////////////////////

/// Evaluator for a JIT-compiled tracing function
///
/// Users are unlikely to use this directly, but it's public because it's an
/// associated type on [`JitFunction`].
struct JitTracingEval<T> {
    choices: VmTrace,
    out: Vec<T>,
}

impl<T> Default for JitTracingEval<T> {
    fn default() -> Self {
        Self {
            choices: VmTrace::default(),
            out: Vec::default(),
        }
    }
}

/// Handle to an owned function pointer for tracing evaluation
pub struct JitTracingFn<T> {
    #[allow(unused)]
    mmap: Mmap,
    choice_count: usize,
    output_count: usize,
    vars: Arc<VarMap>,
    fn_trace: jit_fn!(
        unsafe fn(
            *const T, // vars
            *mut u8,  // choices
            *mut u8,  // simplify (single boolean)
            *mut T,   // output (array)
        )
    ),
}

impl<T> Tape for JitTracingFn<T> {
    type Storage = Mmap;
    fn recycle(self) -> Self::Storage {
        self.mmap
    }

    fn vars(&self) -> &VarMap {
        &self.vars
    }
}

// SAFETY: there is no mutable state in a `JitTracingFn`, and the pointer
// inside of it points to its own `Mmap`, which is owned by an `Arc`
unsafe impl<T> Send for JitTracingFn<T> {}
unsafe impl<T> Sync for JitTracingFn<T> {}

impl<T: From<f32> + Clone> JitTracingEval<T> {
    /// Evaluates a single point, capturing an evaluation trace
    fn eval(
        &mut self,
        tape: &JitTracingFn<T>,
        vars: &[T],
    ) -> (&[T], Option<&VmTrace>) {
        let mut simplify = 0;
        self.choices.resize(tape.choice_count, Choice::Unknown);
        self.choices.fill(Choice::Unknown);
        self.out.resize(tape.output_count, std::f32::NAN.into());
        self.out.fill(f32::NAN.into());
        unsafe {
            (tape.fn_trace)(
                vars.as_ptr(),
                self.choices.as_mut_ptr() as *mut u8,
                &mut simplify,
                self.out.as_mut_ptr(),
            )
        };

        (
            &self.out,
            if simplify != 0 {
                Some(&self.choices)
            } else {
                None
            },
        )
    }
}

/// JIT-based tracing evaluator for interval values
#[derive(Default)]
pub struct JitIntervalEval(JitTracingEval<Interval>);
impl TracingEvaluator for JitIntervalEval {
    type Data = Interval;
    type Tape = JitTracingFn<Interval>;
    type Trace = VmTrace;
    type TapeStorage = Mmap;

    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[Self::Data],
    ) -> Result<(&[Self::Data], Option<&Self::Trace>), Error> {
        tape.vars().check_tracing_arguments(vars)?;
        Ok(self.0.eval(tape, vars))
    }
}

/// JIT-based tracing evaluator for point values
#[derive(Default)]
pub struct JitPointEval(JitTracingEval<f32>);
impl TracingEvaluator for JitPointEval {
    type Data = f32;
    type Tape = JitTracingFn<f32>;
    type Trace = VmTrace;
    type TapeStorage = Mmap;

    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[Self::Data],
    ) -> Result<(&[Self::Data], Option<&Self::Trace>), Error> {
        tape.vars().check_tracing_arguments(vars)?;
        Ok(self.0.eval(tape, vars))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Handle to an owned function pointer for bulk evaluation
pub struct JitBulkFn<T> {
    #[allow(unused)]
    mmap: Mmap,
    vars: Arc<VarMap>,
    fn_bulk: jit_fn!(
        unsafe fn(
            *const *const T, // vars
            *const *mut T,   // out
            u64,             // size
        )
    ),
}

impl<T> Tape for JitBulkFn<T> {
    type Storage = Mmap;
    fn recycle(self) -> Self::Storage {
        self.mmap
    }

    fn vars(&self) -> &VarMap {
        &self.vars
    }
}

/// Maximum SIMD width for any type, checked at runtime (alas)
///
/// We can't use T::SIMD_SIZE directly here due to Rust limitations. Instead we
/// hard-code a maximum SIMD size along with an assertion that should be
/// optimized out; we can't use a constant assertion here due to the same
/// compiler limitations.
const MAX_SIMD_WIDTH: usize = 8;

/// Bulk evaluator for JIT functions
struct JitBulkEval<T> {
    /// Array of pointers used when calling into the JIT function
    input_ptrs: Vec<*const T>,

    /// Array of pointers used when calling into the JIT function
    output_ptrs: Vec<*mut T>,

    /// Scratch array for evaluation of less-than-SIMD-size slices
    scratch: Vec<[T; MAX_SIMD_WIDTH]>,

    /// Output arrays, written to during evaluation
    out: Vec<Vec<T>>,
}

// SAFETY: the pointers in `JitBulkEval` are transient and only scoped to a
// single evaluation.
unsafe impl<T> Sync for JitBulkEval<T> {}
unsafe impl<T> Send for JitBulkEval<T> {}

impl<T> Default for JitBulkEval<T> {
    fn default() -> Self {
        Self {
            out: vec![],
            scratch: vec![],
            input_ptrs: vec![],
            output_ptrs: vec![],
        }
    }
}

// SAFETY: there is no mutable state in a `JitBulkFn`, and the pointer
// inside of it points to its own `Mmap`, which is owned by an `Arc`
unsafe impl<T> Send for JitBulkFn<T> {}
unsafe impl<T> Sync for JitBulkFn<T> {}

impl<T: From<f32> + Copy + SimdSize> JitBulkEval<T> {
    /// Evaluate multiple points
    fn eval<V: std::ops::Deref<Target = [T]>>(
        &mut self,
        tape: &JitBulkFn<T>,
        vars: &[V],
    ) -> BulkOutput<T> {
        let n = vars.first().map(|v| v.deref().len()).unwrap_or(0);

        const OUTPUT_COUNT: usize = 1;
        self.out.resize_with(OUTPUT_COUNT, Vec::new);
        for o in &mut self.out {
            o.resize(n.max(T::SIMD_SIZE), f32::NAN.into());
            o.fill(f32::NAN.into());
        }

        // Special case for when we have fewer items than the native SIMD size,
        // in which case the input slices can't be used as workspace (because
        // they are not valid for the entire range of values read in assembly)
        if n < T::SIMD_SIZE {
            assert!(T::SIMD_SIZE <= MAX_SIMD_WIDTH);

            self.scratch
                .resize(vars.len(), [f32::NAN.into(); MAX_SIMD_WIDTH]);
            for (v, t) in vars.iter().zip(self.scratch.iter_mut()) {
                t[0..n].copy_from_slice(v);
            }

            self.input_ptrs.clear();
            self.input_ptrs
                .extend(self.scratch[..vars.len()].iter().map(|t| t.as_ptr()));

            self.output_ptrs.clear();
            self.output_ptrs
                .extend(self.out.iter_mut().map(|t| t.as_mut_ptr()));

            unsafe {
                (tape.fn_bulk)(
                    self.input_ptrs.as_ptr(),
                    self.output_ptrs.as_ptr(),
                    T::SIMD_SIZE as u64,
                );
            }
        } else {
            // Our vectorized function only accepts sets of a particular width,
            // so we'll find the biggest multiple, then do an extra operation to
            // process any remainders.
            let m = (n / T::SIMD_SIZE) * T::SIMD_SIZE; // Round down
            self.input_ptrs.clear();
            self.input_ptrs.extend(vars.iter().map(|v| v.as_ptr()));

            self.output_ptrs.clear();
            self.output_ptrs
                .extend(self.out.iter_mut().map(|v| v.as_mut_ptr()));
            unsafe {
                (tape.fn_bulk)(
                    self.input_ptrs.as_ptr(),
                    self.output_ptrs.as_ptr(),
                    m as u64,
                );
            }
            // If we weren't given an even multiple of vector width, then we'll
            // handle the remaining items by simply evaluating the *last* full
            // vector in the array again.
            if n != m {
                self.input_ptrs.clear();
                self.output_ptrs.clear();
                unsafe {
                    self.input_ptrs.extend(
                        vars.iter().map(|v| v.as_ptr().add(n - T::SIMD_SIZE)),
                    );
                    self.output_ptrs.extend(
                        self.out
                            .iter_mut()
                            .map(|v| v.as_mut_ptr().add(n - T::SIMD_SIZE)),
                    );
                    (tape.fn_bulk)(
                        self.input_ptrs.as_ptr(),
                        self.output_ptrs.as_ptr(),
                        T::SIMD_SIZE as u64,
                    );
                }
            }
        }
        BulkOutput::new(&self.out, n)
    }
}

/// JIT-based bulk evaluator for arrays of points, yielding point values
#[derive(Default)]
pub struct JitFloatSliceEval(JitBulkEval<f32>);
impl BulkEvaluator for JitFloatSliceEval {
    type Data = f32;
    type Tape = JitBulkFn<Self::Data>;
    type TapeStorage = Mmap;

    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<f32>, Error> {
        tape.vars().check_bulk_arguments(vars)?;
        Ok(self.0.eval(tape, vars))
    }
}

/// JIT-based bulk evaluator for arrays of points, yielding gradient values
#[derive(Default)]
pub struct JitGradSliceEval(JitBulkEval<Grad>);
impl BulkEvaluator for JitGradSliceEval {
    type Data = Grad;
    type Tape = JitBulkFn<Self::Data>;
    type TapeStorage = Mmap;

    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<Grad>, Error> {
        tape.vars().check_bulk_arguments(vars)?;
        Ok(self.0.eval(tape, vars))
    }
}

impl MathFunction for JitFunction {
    fn new(ctx: &Context, nodes: &[Node]) -> Result<Self, Error> {
        GenericVmFunction::new(ctx, nodes).map(JitFunction)
    }
}

/// A [`Shape`](crate::shape::Shape) which uses the JIT evaluator
pub type JitShape = crate::shape::Shape<JitFunction>;

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(JitFunction);
    crate::interval_tests!(JitFunction);
    crate::float_slice_tests!(JitFunction);
    crate::point_tests!(JitFunction);
}
