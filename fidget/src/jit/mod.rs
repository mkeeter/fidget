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
        bulk::BulkEvaluator,
        tracing::TracingEvaluator,
        types::{Grad, Interval},
        EvaluatorStorage, Family,
    },
    jit::mmap::Mmap,
    vm::{ChoiceIndex, ChoiceTape, Choices, Op, Tape, TapeData},
    Error,
};
use dynasmrt::{
    components::PatchLoc, dynasm, AssemblyOffset, DynamicLabel, DynasmApi,
    DynasmError, DynasmLabelApi, TargetKind,
};
use std::ffi::c_void;
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

/// Trait for generating machine assembly
///
/// This is public because it's used to parameterize various other types, but
/// shouldn't be used by clients; indeed, there are no public implementors of
/// this trait.
pub trait AssemblerT {
    /// Initializes the assembler for a threaded code fragment
    fn new() -> Self;

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

    fn build_min_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_max_mem_imm_choice(
        &mut self,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_min_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_max_reg_imm_choice(
        &mut self,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_min_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    );
    fn build_max_mem_reg_choice(
        &mut self,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    );
    fn build_min_reg_reg_choice(
        &mut self,
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    );
    fn build_max_reg_reg_choice(
        &mut self,
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    );
    fn build_copy_imm_reg_choice(
        &mut self,
        out: u8,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_copy_imm_mem_choice(
        &mut self,
        out: u32,
        imm: f32,
        choice: ChoiceIndex,
    );
    fn build_copy_reg_reg_choice(
        &mut self,
        out: u8,
        arg: u8,
        choice: ChoiceIndex,
    );
    fn build_copy_reg_mem_choice(
        &mut self,
        out: u32,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// Builds an entry point for threaded code with the given slot count
    ///
    /// This will likely construct a function prelude, reserve space on the
    /// stack for slot spills, and jump to the first piece of threaded code.
    fn build_entry_point(slot_count: usize) -> Mmap;

    /// Finalize the assembly code, returning a memory-mapped region
    fn finalize(self) -> Result<Mmap, Error>;
}

/// Trait defining SIMD width
pub trait SimdAssembler {
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

/// Build an assembly snippet for the given function
///
/// `t` is expected to be in reverse-evaluation order
fn build_asm_fn<A: AssemblerT>(t: &[Op], slot_count: usize) -> Mmap {
    // This guard may be a unit value on some systems
    #[allow(clippy::let_unit_value)]
    let _guard = Mmap::thread_mode_write();

    let s = Mmap::new(0).unwrap();
    s.make_write();
    let mut asm = A::new();

    for &op in t.iter().rev() {
        match op {
            Op::Load { reg, mem } => {
                asm.build_load(reg, mem);
            }
            Op::Store { reg, mem } => {
                asm.build_store(mem, reg);
            }
            Op::Input { out, input } => {
                asm.build_input(out, input);
            }
            Op::Var { out, var } => {
                asm.build_var(out, var);
            }
            Op::NegReg { out, arg } => {
                asm.build_neg(out, arg);
            }
            Op::AbsReg { out, arg } => {
                asm.build_abs(out, arg);
            }
            Op::RecipReg { out, arg } => {
                asm.build_recip(out, arg);
            }
            Op::SqrtReg { out, arg } => {
                asm.build_sqrt(out, arg);
            }
            Op::CopyReg { out, arg } => {
                asm.build_copy(out, arg);
            }
            Op::SquareReg { out, arg } => {
                asm.build_square(out, arg);
            }
            Op::AddRegReg { out, lhs, rhs } => {
                asm.build_add(out, lhs, rhs);
            }
            Op::MulRegReg { out, lhs, rhs } => {
                asm.build_mul(out, lhs, rhs);
            }
            Op::DivRegReg { out, lhs, rhs } => {
                asm.build_div(out, lhs, rhs);
            }
            Op::SubRegReg { out, lhs, rhs } => {
                asm.build_sub(out, lhs, rhs);
            }
            Op::MinRegReg { out, lhs, rhs } => {
                asm.build_min(out, lhs, rhs);
            }
            Op::MaxRegReg { out, lhs, rhs } => {
                asm.build_max(out, lhs, rhs);
            }
            Op::AddRegImm { out, arg, imm } => {
                asm.build_add_imm(out, arg, imm);
            }
            Op::MulRegImm { out, arg, imm } => {
                asm.build_mul_imm(out, arg, imm);
            }
            Op::DivRegImm { out, arg, imm } => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, arg, reg);
            }
            Op::DivImmReg { out, arg, imm } => {
                let reg = asm.load_imm(imm);
                asm.build_div(out, reg, arg);
            }
            Op::SubImmReg { out, arg, imm } => {
                asm.build_sub_imm_reg(out, arg, imm);
            }
            Op::SubRegImm { out, arg, imm } => {
                asm.build_sub_reg_imm(out, arg, imm);
            }
            Op::MinRegImm { out, arg, imm } => {
                let reg = asm.load_imm(imm);
                asm.build_min(out, arg, reg);
            }
            Op::MaxRegImm { out, arg, imm } => {
                let reg = asm.load_imm(imm);
                asm.build_max(out, arg, reg);
            }
            Op::MaxMemImmChoice { mem, imm, choice } => {
                asm.build_max_mem_imm_choice(mem, imm, choice);
            }
            Op::MinMemImmChoice { mem, imm, choice } => {
                asm.build_min_mem_imm_choice(mem, imm, choice);
            }
            Op::MaxMemRegChoice { mem, arg, choice } => {
                asm.build_max_mem_reg_choice(mem, arg, choice);
            }
            Op::MinMemRegChoice { mem, arg, choice } => {
                asm.build_min_mem_reg_choice(mem, arg, choice);
            }
            Op::MaxRegRegChoice { reg, arg, choice } => {
                asm.build_max_reg_reg_choice(reg, arg, choice);
            }
            Op::MinRegRegChoice { reg, arg, choice } => {
                asm.build_min_reg_reg_choice(reg, arg, choice);
            }
            Op::MaxRegImmChoice { reg, imm, choice } => {
                asm.build_max_reg_imm_choice(reg, imm, choice);
            }
            Op::MinRegImmChoice { reg, imm, choice } => {
                asm.build_min_reg_imm_choice(reg, imm, choice);
            }
            Op::CopyImmRegChoice { out, imm, choice } => {
                asm.build_copy_imm_reg_choice(out, imm, choice);
            }
            Op::CopyImmMemChoice { out, imm, choice } => {
                asm.build_copy_imm_mem_choice(out, imm, choice);
            }
            Op::CopyRegRegChoice { out, arg, choice } => {
                asm.build_copy_reg_reg_choice(out, arg, choice);
            }
            Op::CopyRegMemChoice { out, arg, choice } => {
                asm.build_copy_reg_mem_choice(out, arg, choice);
            }
            Op::CopyImm { out, imm } => {
                let reg = asm.load_imm(imm);
                asm.build_copy(out, reg);
            }
        }
    }

    asm.finalize().expect("failed to build JIT function")
    // JIT execute mode is restored here when the _guard is dropped
}

/// Container for a bunch of JIT code
struct MmapSet {
    point: Mmap,
    interval: Mmap,
    float_slice: Mmap,
    grad_slice: Mmap,
}

trait GetPointer<T> {
    fn get_pointer(&self) -> *const c_void;
}

impl GetPointer<f32> for MmapSet {
    fn get_pointer(&self) -> *const c_void {
        self.point.as_slice().as_ptr() as *const c_void
    }
}

impl GetPointer<Interval> for MmapSet {
    fn get_pointer(&self) -> *const c_void {
        self.interval.as_slice().as_ptr() as *const c_void
    }
}

impl GetPointer<*const f32> for MmapSet {
    fn get_pointer(&self) -> *const c_void {
        self.float_slice.as_slice().as_ptr() as *const c_void
    }
}

impl GetPointer<*const Grad> for MmapSet {
    fn get_pointer(&self) -> *const c_void {
        self.grad_slice.as_slice().as_ptr() as *const c_void
    }
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

    /// The root tape contains entry points for threaded code
    type TapeData = MmapSet;

    /// Each group contains a chunk of threaded code to do its evaluation
    type GroupMetadata = MmapSet;

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

    fn build(
        slot_count: usize,
        tapes: &[ChoiceTape],
    ) -> (Self::TapeData, Vec<Self::GroupMetadata>) {
        let mut out = vec![];
        for t in tapes {
            let point = build_asm_fn::<point::PointAssembler>(
                t.tape.as_slice(),
                slot_count,
            );
            let interval = build_asm_fn::<interval::IntervalAssembler>(
                t.tape.as_slice(),
                slot_count,
            );
            let float_slice = build_asm_fn::<float_slice::FloatSliceAssembler>(
                t.tape.as_slice(),
                slot_count,
            );
            let grad_slice = build_asm_fn::<grad_slice::GradSliceAssembler>(
                t.tape.as_slice(),
                slot_count,
            );
            out.push(MmapSet {
                point,
                interval,
                float_slice,
                grad_slice,
            })
        }
        let point = point::PointAssembler::build_entry_point(slot_count);
        let interval =
            interval::IntervalAssembler::build_entry_point(slot_count);
        let float_slice =
            float_slice::FloatSliceAssembler::build_entry_point(slot_count);
        let grad_slice =
            grad_slice::GradSliceAssembler::build_entry_point(slot_count);
        (
            MmapSet {
                point,
                interval,
                float_slice,
                grad_slice,
            },
            out,
        )
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

/// Handle owning a JIT-compiled function of some kind
///
/// Users are unlikely to use this directly; consider using the
/// [`jit::Eval`](Eval) evaluator family instead.
pub struct JitEval<T> {
    code: ThreadedCode,
    var_count: usize,
    _p: std::marker::PhantomData<fn() -> T>,
}

impl<I: AssemblerT> Clone for JitEval<I> {
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(), // TODO: this is expensive
            var_count: self.var_count,
            _p: std::marker::PhantomData,
        }
    }
}

// SAFETY: there is no mutable state in a `JitEval`, and the pointer
// inside of it points to its own `Mmap`, which is owned by an `Arc`
unsafe impl<I: Family> Send for JitEval<I> {}
unsafe impl<I: Family> Sync for JitEval<I> {}

/// Threaded code, which is a set of pointers to JIT-compiled code
///
/// Each chunk of code must end with a jump to the next one
#[derive(Clone, Default)]
struct ThreadedCode {
    entry_points: Arc<Vec<*const c_void>>,

    /// Handle to the parent `TapeData`
    ///
    /// This ensures that [`entry_points`] remains valid for the lifetime of
    /// a given `ThreadedCode` instance.
    data: Option<Arc<TapeData<Eval>>>,
}

// SAFETY: there is no mutable state in a `ThreadedCode`, and it owns a strong
// reference to the memory into which it contains pointers.
unsafe impl Send for ThreadedCode {}
unsafe impl Sync for ThreadedCode {}

impl<T> EvaluatorStorage<Eval> for JitEval<T>
where
    MmapSet: GetPointer<T>,
{
    type Storage = ThreadedCode;
    fn new_with_storage(t: &Tape<Eval>, mut prev: Self::Storage) -> Self {
        prev.entry_points.clear();
        prev.data = Some(t.data().clone());
        for g in t.active_groups().iter().rev() {
            prev.entry_points
                .push(<MmapSet as GetPointer<T>>::get_pointer(&t.data().data));
        }
        // TODO: push finalize here
        Self {
            code: prev,
            var_count: t.var_count(),
            _p: std::marker::PhantomData,
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.code.entry_points)
            .ok()
            .map(|entry_points| ThreadedCode {
                entry_points: Arc::new(entry_points), // TODO allocation :(
                data: None,
            })
    }
}

impl<T> TracingEvaluator<T, Eval> for JitEval<T>
where
    MmapSet: GetPointer<T>,
{
    type Data = ();

    /// Evaluates a single point, capturing execution in `choices`
    fn eval_with(
        &self,
        x: T,
        y: T,
        z: T,
        vars: &[f32],
        choices: &mut Choices,
        _data: &mut (),
    ) -> (T, bool) {
        let mut simplify = 0;
        assert_eq!(vars.len(), self.var_count);
        let start = <MmapSet as GetPointer<T>>::get_pointer(
            &self.code.data.unwrap().data,
        );

        // SAFETY: this is a pointer to a hand-written entry point in memory
        // that's owned by self.code.data
        let f: jit_fn!(
            unsafe fn(
                T,          // X
                T,          // Y
                T,          // Z
                *const f32, // vars
                *mut u64,   // choices
                *mut u8,    // simplify (single boolean)
            ) -> T
        ) = unsafe { std::mem::transmute(start) };

        let out = unsafe {
            f(
                x,
                y,
                z,
                vars.as_ptr(),
                choices.as_mut().as_mut_ptr(),
                &mut simplify,
            )
        };
        (out, simplify != 0)
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<T, I: SimdAssembler> BulkEvaluator<T, Eval> for JitEval<I>
where
    T: Copy + From<f32>,
    MmapSet: GetPointer<*const T>,
{
    type Data = ();

    /// Evaluate multiple points
    fn eval_with(
        &self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [T],
        choices: &mut Choices,
        _data: &mut (),
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.var_count);

        let n = xs.len();

        let start = <MmapSet as GetPointer<*const T>>::get_pointer(
            &self.code.data.unwrap().data,
        );

        let f: jit_fn!(
            unsafe fn(
                *const *const c_void,
                *const f32, // X
                *const f32, // Y
                *const f32, // Z
                *const f32, // vars
                *mut T,     // out
                u64,        // size
            ) -> T
        ) = unsafe { std::mem::transmute(start) };

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
                f(
                    self.code.entry_points.as_ptr(),
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
                f(
                    self.code.entry_points.as_ptr(),
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
                    f(
                        self.code.entry_points.as_ptr(),
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
