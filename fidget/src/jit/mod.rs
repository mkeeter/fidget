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
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, VecAssembler};
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

/// Scratch register used in operations that use memory directly
const SCRATCH_REG: u8 = arch::SCRATCH_REG;

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
pub trait AssemblerT<
    D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>,
>
{
    /// Data type that is operated on (used for computing stack offsets)
    type T;

    /// Returns the offset of the given stack location
    fn stack_pos(slot: u32) -> u32 {
        arch::stack_pos::<Self::T>(slot)
    }

    /// Assembles a function prelude into the given buffer
    fn function_entry(ops: &mut D, slot_count: usize) -> u32 {
        arch::function_entry::<Self::T, _>(ops, slot_count)
    }

    /// Builds an entry point for threaded code with the given slot count
    ///
    /// This will likely construct a function prelude, reserve space on the
    /// stack for slot spills, and jump to the first piece of threaded code.
    ///
    /// `choice_array_size` is the number of `u64` words in the choice array
    fn build_entry_point(
        ops: &mut D,
        slot_count: usize,
        choice_array_size: usize,
    ) -> usize;

    /// Builds a load from memory to a register
    fn build_load(ops: &mut D, dst_reg: u8, src_mem: u32);

    /// Builds a store from a register to a memory location
    fn build_store(ops: &mut D, dst_mem: u32, src_reg: u8);

    /// Copies the given input to `out_reg`
    fn build_input(ops: &mut D, out_reg: u8, src_arg: u8);

    /// Copies a variable (provided in an input array) to `out_reg`
    fn build_var(ops: &mut D, out_reg: u8, src_arg: u32);

    /// Copies a register
    fn build_copy(ops: &mut D, out_reg: u8, lhs_reg: u8);

    /// Unary negation
    fn build_neg(ops: &mut D, out_reg: u8, lhs_reg: u8);

    /// Absolute value
    fn build_abs(ops: &mut D, out_reg: u8, lhs_reg: u8);

    /// Reciprocal (1 / `lhs_reg`)
    fn build_recip(ops: &mut D, out_reg: u8, lhs_reg: u8);

    /// Square root
    fn build_sqrt(ops: &mut D, out_reg: u8, lhs_reg: u8);

    /// Square
    ///
    /// This has a default implementation, but can be overloaded for efficiency;
    /// for example, in interval arithmetic, we benefit from knowing that both
    /// values are the same.
    fn build_square(ops: &mut D, out_reg: u8, lhs_reg: u8) {
        Self::build_mul(ops, out_reg, lhs_reg, lhs_reg)
    }

    /// Addition
    fn build_add(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Subtraction
    fn build_sub(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Multiplication
    fn build_mul(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Division
    fn build_div(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Maximum of two values
    ///
    /// In a tracing evaluator, this function must also write to the `choices`
    /// array and may set `simplify` if one branch is always taken.
    fn build_max(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    /// Minimum of two values
    ///
    /// In a tracing evaluator, this function must also write to the `choices`
    /// array and may set `simplify` if one branch is always taken.
    fn build_min(ops: &mut D, out_reg: u8, lhs_reg: u8, rhs_reg: u8);

    // Special-case functions for immediates.  In some cases, you can be more
    // efficient if you know that an argument is an immediate (for example, both
    // values in the interval will be the same, and it wlll have no gradients).

    /// Builds a addition (immediate + register)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_add_imm(ops: &mut D, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = Self::load_imm(ops, imm);
        Self::build_add(ops, out_reg, lhs_reg, imm);
    }
    /// Builds a subtraction (immediate − register)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_sub_imm_reg(ops: &mut D, out_reg: u8, arg: u8, imm: f32) {
        let imm = Self::load_imm(ops, imm);
        Self::build_sub(ops, out_reg, imm, arg);
    }
    /// Builds a subtraction (register − immediate)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_sub_reg_imm(ops: &mut D, out_reg: u8, arg: u8, imm: f32) {
        let imm = Self::load_imm(ops, imm);
        Self::build_sub(ops, out_reg, arg, imm);
    }
    /// Builds a multiplication (register × immediate)
    ///
    /// This has a default implementation, but can be overloaded for efficiency
    fn build_mul_imm(ops: &mut D, out_reg: u8, lhs_reg: u8, imm: f32) {
        let imm = Self::load_imm(ops, imm);
        Self::build_mul(ops, out_reg, lhs_reg, imm);
    }

    /// Loads an immediate into a register, returning that register
    fn load_imm(ops: &mut D, imm: f32) -> u8;

    /// `min` operation with an `inout` memory address, immediate, and choice
    fn build_min_mem_imm_choice(
        ops: &mut D,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// `max` operation with an `inout` memory address, immediate, and choice
    fn build_max_mem_imm_choice(
        ops: &mut D,
        mem: u32,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// `min` operation with an `inout` register, immediate, and choice
    fn build_min_reg_imm_choice(
        ops: &mut D,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// `max` operation with an `inout` register, immediate, and choice
    fn build_max_reg_imm_choice(
        ops: &mut D,
        reg: u8,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// `min` operation with an `inout` memory address, register, and choice
    fn build_min_mem_reg_choice(
        ops: &mut D,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// `max` operation with an `inout` memory address, register, and choice
    fn build_max_mem_reg_choice(
        ops: &mut D,
        mem: u32,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// `min` operation with an `inout` and argument register, and choice
    fn build_min_reg_reg_choice(
        ops: &mut D,
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// `max` operation with an `inout` and argument register, and choice
    fn build_max_reg_reg_choice(
        ops: &mut D,
        reg: u8,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// Copy an immediate to a register and set the given choice bit
    fn build_copy_imm_reg_choice(
        ops: &mut D,
        out: u8,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// Copy an immediate to a memory address and set the given choice bit
    fn build_copy_imm_mem_choice(
        ops: &mut D,
        out: u32,
        imm: f32,
        choice: ChoiceIndex,
    );

    /// Copy a register to a register and set the given choice bit
    fn build_copy_reg_reg_choice(
        ops: &mut D,
        out: u8,
        arg: u8,
        choice: ChoiceIndex,
    );

    /// Copy a register to a memory address and set the given choice bit
    fn build_copy_reg_mem_choice(
        ops: &mut D,
        out: u32,
        arg: u8,
        choice: ChoiceIndex,
    );
}

/// Trait defining SIMD width
pub trait SimdType {
    /// Number of elements processed in a single iteration
    ///
    /// This value is used when checking array sizes, as we want to be sure to
    /// pass the JIT code an appropriately sized array.
    const SIMD_SIZE: usize;
}

////////////////////////////////////////////////////////////////////////////////

/// Build an assembly snippet for the given function
///
/// `t` is expected to be in reverse-evaluation order
fn build_asm_fn<
    A: AssemblerT<D>,
    D: DynasmApi + DynasmLabelApi<Relocation = arch::Relocation>,
>(
    ops: &mut D,
    t: &[Op],
) -> usize {
    let out = ops.offset().0;
    for &op in t.iter().rev() {
        match op {
            Op::Load { reg, mem } => {
                A::build_load(ops, reg, mem);
            }
            Op::Store { reg, mem } => {
                A::build_store(ops, mem, reg);
            }
            Op::Input { out, input } => {
                A::build_input(ops, out, input);
            }
            Op::Var { out, var } => {
                A::build_var(ops, out, var);
            }
            Op::NegReg { out, arg } => {
                A::build_neg(ops, out, arg);
            }
            Op::AbsReg { out, arg } => {
                A::build_abs(ops, out, arg);
            }
            Op::RecipReg { out, arg } => {
                A::build_recip(ops, out, arg);
            }
            Op::SqrtReg { out, arg } => {
                A::build_sqrt(ops, out, arg);
            }
            Op::CopyReg { out, arg } => {
                A::build_copy(ops, out, arg);
            }
            Op::SquareReg { out, arg } => {
                A::build_square(ops, out, arg);
            }
            Op::AddRegReg { out, lhs, rhs } => {
                A::build_add(ops, out, lhs, rhs);
            }
            Op::MulRegReg { out, lhs, rhs } => {
                A::build_mul(ops, out, lhs, rhs);
            }
            Op::DivRegReg { out, lhs, rhs } => {
                A::build_div(ops, out, lhs, rhs);
            }
            Op::SubRegReg { out, lhs, rhs } => {
                A::build_sub(ops, out, lhs, rhs);
            }
            Op::MinRegReg { out, lhs, rhs } => {
                A::build_min(ops, out, lhs, rhs);
            }
            Op::MaxRegReg { out, lhs, rhs } => {
                A::build_max(ops, out, lhs, rhs);
            }
            Op::AddRegImm { out, arg, imm } => {
                A::build_add_imm(ops, out, arg, imm);
            }
            Op::MulRegImm { out, arg, imm } => {
                A::build_mul_imm(ops, out, arg, imm);
            }
            Op::DivRegImm { out, arg, imm } => {
                let reg = A::load_imm(ops, imm);
                A::build_div(ops, out, arg, reg);
            }
            Op::DivImmReg { out, arg, imm } => {
                let reg = A::load_imm(ops, imm);
                A::build_div(ops, out, reg, arg);
            }
            Op::SubImmReg { out, arg, imm } => {
                A::build_sub_imm_reg(ops, out, arg, imm);
            }
            Op::SubRegImm { out, arg, imm } => {
                A::build_sub_reg_imm(ops, out, arg, imm);
            }
            Op::MinRegImm { out, arg, imm } => {
                let reg = A::load_imm(ops, imm);
                A::build_min(ops, out, arg, reg);
            }
            Op::MaxRegImm { out, arg, imm } => {
                let reg = A::load_imm(ops, imm);
                A::build_max(ops, out, arg, reg);
            }
            Op::MaxMemImmChoice { mem, imm, choice } => {
                A::build_max_mem_imm_choice(ops, mem, imm, choice);
            }
            Op::MinMemImmChoice { mem, imm, choice } => {
                A::build_min_mem_imm_choice(ops, mem, imm, choice);
            }
            Op::MaxMemRegChoice { mem, arg, choice } => {
                A::build_max_mem_reg_choice(ops, mem, arg, choice);
            }
            Op::MinMemRegChoice { mem, arg, choice } => {
                A::build_min_mem_reg_choice(ops, mem, arg, choice);
            }
            Op::MaxRegRegChoice { reg, arg, choice } => {
                A::build_max_reg_reg_choice(ops, reg, arg, choice);
            }
            Op::MinRegRegChoice { reg, arg, choice } => {
                A::build_min_reg_reg_choice(ops, reg, arg, choice);
            }
            Op::MaxRegImmChoice { reg, imm, choice } => {
                A::build_max_reg_imm_choice(ops, reg, imm, choice);
            }
            Op::MinRegImmChoice { reg, imm, choice } => {
                A::build_min_reg_imm_choice(ops, reg, imm, choice);
            }
            Op::CopyImmRegChoice { out, imm, choice } => {
                A::build_copy_imm_reg_choice(ops, out, imm, choice);
            }
            Op::CopyImmMemChoice { out, imm, choice } => {
                A::build_copy_imm_mem_choice(ops, out, imm, choice);
            }
            Op::CopyRegRegChoice { out, arg, choice } => {
                A::build_copy_reg_reg_choice(ops, out, arg, choice);
            }
            Op::CopyRegMemChoice { out, arg, choice } => {
                A::build_copy_reg_mem_choice(ops, out, arg, choice);
            }
            Op::CopyImm { out, imm } => {
                let reg = A::load_imm(ops, imm);
                A::build_copy(ops, out, reg);
            }
        }
    }
    arch::build_jump(ops);
    out
}

/// Container for a bunch of JIT code
pub struct MmapOffsets {
    point: usize,
    interval: usize,
    float_slice: usize,
    grad_slice: usize,
}

trait GetPointer<T> {
    fn get_pointer(&self, mmap: &Mmap) -> *const c_void;
}

impl GetPointer<f32> for MmapOffsets {
    fn get_pointer(&self, mmap: &Mmap) -> *const c_void {
        mmap.as_ptr().wrapping_add(self.point) as *const c_void
    }
}

impl GetPointer<Interval> for MmapOffsets {
    fn get_pointer(&self, mmap: &Mmap) -> *const c_void {
        mmap.as_ptr().wrapping_add(self.interval) as *const c_void
    }
}

impl GetPointer<*const f32> for MmapOffsets {
    fn get_pointer(&self, mmap: &Mmap) -> *const c_void {
        mmap.as_ptr().wrapping_add(self.float_slice) as *const c_void
    }
}

impl GetPointer<*const Grad> for MmapOffsets {
    fn get_pointer(&self, mmap: &Mmap) -> *const c_void {
        mmap.as_ptr().wrapping_add(self.grad_slice) as *const c_void
    }
}

/// Associated data for a tape being evaluated by a JIT evaluator
///
/// This is a heavy-weight structure that is kept in one place.
pub struct JitData {
    /// Raw assembled code
    mmap: Mmap,

    /// Offset of a return statement that can be jumped into
    ///
    /// This is an offset into [`self.mmap`]
    ret_offset: usize,

    /// Function which jumps into the threaded code
    ///
    /// These are specified as offsets into [`self.mmap`]
    trampolines: MmapOffsets,
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

    /// The root tape contains entry and exit points for threaded code
    type TapeData = JitData;

    /// Each group contains a chunk of threaded code to do its evaluation
    type GroupMetadata = MmapOffsets;

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
        choice_array_size: usize,
        tapes: &[ChoiceTape],
    ) -> (Self::TapeData, Vec<Self::GroupMetadata>) {
        let mut data = VecAssembler::new(0);
        let point = point::PointAssembler::build_entry_point(
            &mut data,
            slot_count,
            choice_array_size,
        );
        let mut points = vec![];
        for t in tapes {
            points.push(build_asm_fn::<point::PointAssembler, _>(
                &mut data,
                t.tape.as_slice(),
            ));
        }

        let interval = interval::IntervalAssembler::build_entry_point(
            &mut data,
            slot_count,
            choice_array_size,
        );
        let mut intervals = vec![];
        for t in tapes {
            intervals.push(build_asm_fn::<interval::IntervalAssembler, _>(
                &mut data,
                t.tape.as_slice(),
            ));
        }

        let float_slice = float_slice::FloatSliceAssembler::build_entry_point(
            &mut data,
            slot_count,
            choice_array_size,
        );
        let mut float_slices = vec![];
        for t in tapes {
            float_slices.push(build_asm_fn::<
                float_slice::FloatSliceAssembler,
                _,
            >(&mut data, t.tape.as_slice()));
        }

        let grad_slice = grad_slice::GradSliceAssembler::build_entry_point(
            &mut data,
            slot_count,
            choice_array_size,
        );
        let mut grad_slices = vec![];
        for t in tapes {
            grad_slices.push(
                build_asm_fn::<grad_slice::GradSliceAssembler, _>(
                    &mut data,
                    t.tape.as_slice(),
                ),
            );
        }

        let mut out = vec![];
        for i in 0..points.len() {
            out.push(MmapOffsets {
                point: points[i],
                interval: intervals[i],
                float_slice: float_slices[i],
                grad_slice: grad_slices[i],
            })
        }

        // Build a tiny assembler with a return statement
        let ret_offset = data.offset().0;
        dynasm!(data ; ret );

        (
            JitData {
                trampolines: MmapOffsets {
                    point,
                    interval,
                    float_slice,
                    grad_slice,
                },
                ret_offset,
                mmap: data.try_into().unwrap(),
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
    code: Arc<ThreadedCode>,
    var_count: usize,
    _p: std::marker::PhantomData<fn() -> T>,
}

impl<I> Clone for JitEval<I> {
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
unsafe impl<I> Send for JitEval<I> {}
unsafe impl<I> Sync for JitEval<I> {}

/// Threaded code, which is a set of pointers to JIT-compiled code
///
/// Each chunk of code must end with a jump to the next one
#[derive(Clone)]
struct ThreadedCode {
    entry_points: Vec<*const c_void>,

    /// Handle to the parent `TapeData`
    ///
    /// This ensures that [`entry_points`] remains valid for the lifetime of
    /// a given `ThreadedCode` instance.
    tape_data: Arc<TapeData<Eval>>,
}

impl<T> EvaluatorStorage<Eval> for JitEval<T>
where
    MmapOffsets: GetPointer<T>,
{
    type Storage = Vec<*const c_void>;
    fn new_with_storage(t: &Tape<Eval>, mut prev: Self::Storage) -> Self {
        prev.clear();
        for g in t.active_groups().iter().rev() {
            prev.push(<MmapOffsets as GetPointer<T>>::get_pointer(
                &t.data().groups[*g].data,
                &t.data().data.mmap,
            ));
        }
        prev.push(
            t.data()
                .data
                .mmap
                .as_ptr()
                .wrapping_add(t.data().data.ret_offset) as *const _,
        );
        let prev = ThreadedCode {
            entry_points: prev,
            tape_data: t.data().clone(),
        };
        // TODO: push finalize here
        Self {
            code: Arc::new(prev),
            var_count: t.var_count(),
            _p: std::marker::PhantomData,
        }
    }

    fn take(self) -> Option<Self::Storage> {
        Arc::try_unwrap(self.code).ok().map(|v| v.entry_points)
    }
}

impl<T> TracingEvaluator<T, Eval> for JitEval<T>
where
    MmapOffsets: GetPointer<T>,
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
        let trampoline = <MmapOffsets as GetPointer<T>>::get_pointer(
            &self.code.tape_data.data.trampolines,
            &self.code.tape_data.data.mmap,
        );

        // SAFETY: this is a pointer to a hand-written entry point in memory
        // that's owned by self.code.data
        let f: jit_fn!(
            unsafe fn(
                *const *const c_void,
                T,          // X
                T,          // Y
                T,          // Z
                *const f32, // vars
                *mut u64,   // choices
                *mut u32,   // simplify (single word)
            ) -> T
        ) = unsafe { std::mem::transmute(trampoline) };

        let initial_entry_point = self.code.entry_points.as_ptr();

        let out = unsafe {
            f(
                initial_entry_point,
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

impl<T, I: SimdType> BulkEvaluator<T, Eval> for JitEval<I>
where
    T: Copy + From<f32>,
    MmapOffsets: GetPointer<*const T>,
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

        let trampoline = <MmapOffsets as GetPointer<*const T>>::get_pointer(
            &self.code.tape_data.data.trampolines,
            &self.code.tape_data.data.mmap,
        );

        let f: jit_fn!(
            unsafe fn(
                *const *const c_void, // data
                *const f32,           // X
                *const f32,           // Y
                *const f32,           // Z
                *const f32,           // vars
                *mut T,               // out
                u64,                  // size
                *mut u64,             // choices
            ) -> T
        ) = unsafe { std::mem::transmute(trampoline) };

        let initial_entry_point = self.code.entry_points.as_ptr();

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
                    initial_entry_point,
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vars.as_ptr(),
                    tmp.as_mut_ptr(),
                    I::SIMD_SIZE as u64,
                    choices.as_mut().as_mut_ptr(),
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
                    initial_entry_point,
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    vars.as_ptr(),
                    out.as_mut_ptr(),
                    m as u64,
                    choices.as_mut().as_mut_ptr(),
                );
            }
            // If we weren't given an even multiple of vector width, then we'll
            // handle the remaining items by simply evaluating the *last* full
            // vector in the array again.
            if n != m {
                unsafe {
                    f(
                        initial_entry_point,
                        xs.as_ptr().add(n - I::SIMD_SIZE),
                        ys.as_ptr().add(n - I::SIMD_SIZE),
                        zs.as_ptr().add(n - I::SIMD_SIZE),
                        vars.as_ptr(),
                        out.as_mut_ptr().add(n - I::SIMD_SIZE),
                        I::SIMD_SIZE as u64,
                        choices.as_mut().as_mut_ptr(),
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
