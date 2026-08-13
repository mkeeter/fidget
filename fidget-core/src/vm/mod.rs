//! Simple virtual machine for shape evaluation
use crate::{
    Context,
    compiler::RegOp,
    context::{BadNode, Node},
    eval::{
        BulkEvalError, BulkEvaluator, BulkOutput, Function, MathFunction, Tape,
        Trace, TracingEvalError, TracingEvaluator,
    },
    render::{RenderHints, TileSizes},
    shape::Shape,
    types::{FloatExt, Grad, Interval},
    var::VarMap,
};
use std::sync::Arc;

mod choice;
mod data;

pub use choice::Choice;
use data::BadChoiceSlice;
pub use data::{VmData, VmWorkspace};

////////////////////////////////////////////////////////////////////////////////

/// Function which uses the VM backend for evaluation
///
/// Internally, the [`VmFunction`] stores an [`Arc<VmData>`](VmData), and
/// iterates over a [`Vec<RegOp>`](RegOp) to perform evaluation.
///
/// All of the associated [`Tape`] types simply clone the internal `Arc`;
/// there's no separate planning required to generate a tape.
pub type VmFunction = GenericVmFunction<{ u8::MAX as usize }>;

/// Shape that uses the [`VmFunction`] backend for evaluation
pub type VmShape = Shape<VmFunction>;

/// Tape storage type which indicates that there's no actual backing storage
#[derive(Default)]
pub struct EmptyTapeStorage;

/// Tape which uses the VM backend for evaluation
///
/// This tape type is equivalent to a [`GenericVmFunction`], but implements
/// different traits ([`Tape`] instead of [`Function`]).
#[derive(Clone)]
pub struct GenericVmTape<const N: usize>(Arc<VmData<N>>);

impl<const N: usize> GenericVmTape<N> {
    /// Returns a handle to the inner [`VmData`] used by the tape
    pub fn data(&self) -> &VmData<N> {
        &self.0
    }
}

impl<const N: usize> Tape for GenericVmTape<N> {
    type Storage = EmptyTapeStorage;
    fn recycle(self) -> Option<Self::Storage> {
        Some(EmptyTapeStorage)
    }

    fn vars(&self) -> &VarMap {
        &self.0.vars
    }

    fn output_count(&self) -> usize {
        self.0.output_count()
    }
}

/// A trace captured by a VM evaluation
///
/// This is a thin wrapper around a [`Vec<Choice>`](Choice).
#[derive(Clone, Default, Eq, PartialEq)]
pub struct VmTrace(Vec<Choice>);

impl VmTrace {
    /// Fills the trace with the given value
    pub fn fill(&mut self, v: Choice) {
        self.0.fill(v);
    }
    /// Resizes the trace, using the new value if it needs to be extended
    pub fn resize(&mut self, n: usize, v: Choice) {
        self.0.resize(n, v);
    }
    /// Returns the inner choice slice
    pub fn as_slice(&self) -> &[Choice] {
        self.0.as_slice()
    }
    /// Returns the inner choice slice as a mutable reference
    pub fn as_mut_slice(&mut self) -> &mut [Choice] {
        self.0.as_mut_slice()
    }
    /// Returns a pointer to the allocated choice array
    pub fn as_mut_ptr(&mut self) -> *mut Choice {
        self.0.as_mut_ptr()
    }
}

impl Trace for VmTrace {
    fn copy_from(&mut self, other: &VmTrace) {
        self.0.resize(other.0.len(), Choice::Unknown);
        self.0.copy_from_slice(&other.0);
    }
}

#[cfg(any(test, feature = "eval-tests"))]
impl From<Vec<Choice>> for VmTrace {
    fn from(v: Vec<Choice>) -> Self {
        Self(v)
    }
}

#[cfg(any(test, feature = "eval-tests"))]
impl AsRef<[Choice]> for VmTrace {
    fn as_ref(&self) -> &[Choice] {
        &self.0
    }
}

/// VM-backed shape with a configurable number of registers
///
/// You are unlikely to use this directly; [`VmShape`] should be used for
/// VM-based evaluation.
#[derive(Clone)]
pub struct GenericVmFunction<const N: usize>(Arc<VmData<N>>);

impl<const N: usize> From<VmData<N>> for GenericVmFunction<N> {
    fn from(d: VmData<N>) -> Self {
        Self(d.into())
    }
}

impl<const N: usize> GenericVmFunction<N> {
    /// Returns a characteristic size (the length of the inner assembly tape)
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Reclaim the inner `VmData` if there's only a single reference
    pub fn recycle(self) -> Option<VmData<N>> {
        Arc::try_unwrap(self.0).ok()
    }

    /// Borrows the inner [`VmData`]
    pub fn data(&self) -> &VmData<N> {
        self.0.as_ref()
    }

    /// Returns a [`GenericVmTape`] for the given function
    pub fn tape(&self) -> GenericVmTape<N> {
        GenericVmTape(self.0.clone())
    }

    /// Returns the number of choices (i.e. `min` and `max` nodes) in the tape
    pub fn choice_count(&self) -> usize {
        self.0.choice_count()
    }

    /// Returns the number of outputs in the tape
    pub fn output_count(&self) -> usize {
        self.0.output_count()
    }

    /// Simplifies the function with the given trace and a new register count
    pub fn simplify_with<const M: usize>(
        &self,
        trace: &VmTrace,
        storage: VmData<M>,
        workspace: &mut VmWorkspace<M>,
    ) -> Result<GenericVmFunction<M>, BadTrace> {
        let d = self.0.simplify::<M>(trace.as_slice(), workspace, storage)?;
        Ok(GenericVmFunction(Arc::new(d)))
    }
}

/// Error type for simplification
#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct BadTrace(#[from] pub BadChoiceSlice);

impl<const N: usize> Function for GenericVmFunction<N> {
    type Storage = VmData<N>;
    type Workspace = VmWorkspace<N>;

    type TapeStorage = EmptyTapeStorage;

    type FloatSliceEval = VmFloatSliceEval<N>;
    type GradSliceEval = VmGradSliceEval<N>;
    type PointEval = VmPointEval<N>;
    type IntervalEval = VmIntervalEval<N>;
    type Trace = VmTrace;

    #[inline]
    fn float_slice_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn grad_slice_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn point_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn interval_tape(&self, _storage: EmptyTapeStorage) -> GenericVmTape<N> {
        self.tape()
    }

    #[inline]
    fn simplify(
        &self,
        trace: &Self::Trace,
        storage: Self::Storage,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, BadTrace> {
        self.simplify_with(trace, storage, workspace)
    }

    #[inline]
    fn recycle(self) -> Option<Self::Storage> {
        GenericVmFunction::recycle(self)
    }

    #[inline]
    fn size(&self) -> usize {
        GenericVmFunction::size(self)
    }

    #[inline]
    fn vars(&self) -> &VarMap {
        &self.0.vars
    }

    #[inline]
    fn can_simplify(&self) -> bool {
        self.0.choice_count() > 0
    }

    #[inline]
    fn output_count(&self) -> usize {
        self.0.output_count()
    }
}

impl<const N: usize> RenderHints for GenericVmFunction<N> {
    fn tile_sizes_3d() -> TileSizes {
        TileSizes::new(&[128, 64, 32, 16, 8]).unwrap()
    }

    fn tile_sizes_2d() -> TileSizes {
        TileSizes::new(&[128, 32, 8]).unwrap()
    }
}

impl<const N: usize> MathFunction for GenericVmFunction<N> {
    fn new(ctx: &Context, nodes: &[Node]) -> Result<Self, BadNode> {
        let d = VmData::new(ctx, nodes)?;
        Ok(Self(d.into()))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Helper struct to reduce boilerplate conversions
struct SlotArray<'a, T>(&'a mut [T]);
impl<T> std::ops::Index<u8> for SlotArray<'_, T> {
    type Output = T;
    fn index(&self, i: u8) -> &Self::Output {
        &self.0[i as usize]
    }
}
impl<T> std::ops::IndexMut<u8> for SlotArray<'_, T> {
    fn index_mut(&mut self, i: u8) -> &mut T {
        &mut self.0[i as usize]
    }
}
impl<T> std::ops::Index<u32> for SlotArray<'_, T> {
    type Output = T;
    fn index(&self, i: u32) -> &Self::Output {
        &self.0[i as usize]
    }
}
impl<T> std::ops::IndexMut<u32> for SlotArray<'_, T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        &mut self.0[i as usize]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Generic VM evaluator for tracing evaluation
struct TracingVmEval<T> {
    slots: Vec<T>,
    out: Vec<T>,
    choices: VmTrace,
}

impl<T> Default for TracingVmEval<T> {
    fn default() -> Self {
        Self {
            slots: Vec::default(),
            out: Vec::default(),
            choices: VmTrace::default(),
        }
    }
}

impl<T: From<f32> + Clone> TracingVmEval<T> {
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>) {
        self.slots.resize(tape.slot_count(), f32::NAN.into());
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.out.resize(tape.output_count(), f32::NAN.into());
        self.choices.fill(Choice::Unknown);
    }
}

/// VM-based tracing evaluator for intervals
#[derive(Default)]
pub struct VmIntervalEval<const N: usize>(TracingVmEval<Interval>);
impl<const N: usize> TracingEvaluator for VmIntervalEval<N> {
    type Data = Interval;
    type Tape = GenericVmTape<N>;
    type Trace = VmTrace;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[Interval],
    ) -> Result<(&[Interval], Option<&VmTrace>), TracingEvalError> {
        tape.vars().check_tracing_arguments(vars)?;
        let tape = tape.data();
        self.0.resize_slots(tape);

        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        let mut choices = self.0.choices.as_mut_slice().iter_mut();
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize] = v[arg];
                }
                RegOp::Input(out, i) => {
                    v[out] = vars[i as usize];
                }
                RegOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                RegOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RegOp::RecipReg(out, arg) => {
                    v[out] = v[arg].recip();
                }
                RegOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                RegOp::SquareReg(out, arg) => {
                    v[out] = v[arg].square();
                }
                RegOp::FloorReg(out, arg) => {
                    v[out] = v[arg].floor();
                }
                RegOp::CeilReg(out, arg) => {
                    v[out] = v[arg].ceil();
                }
                RegOp::RoundReg(out, arg) => {
                    v[out] = v[arg].round();
                }
                RegOp::SinReg(out, arg) => {
                    v[out] = v[arg].sin();
                }
                RegOp::CosReg(out, arg) => {
                    v[out] = v[arg].cos();
                }
                RegOp::TanReg(out, arg) => {
                    v[out] = v[arg].tan();
                }
                RegOp::AsinReg(out, arg) => {
                    v[out] = v[arg].asin();
                }
                RegOp::AcosReg(out, arg) => {
                    v[out] = v[arg].acos();
                }
                RegOp::AtanReg(out, arg) => {
                    v[out] = v[arg].atan();
                }
                RegOp::ExpReg(out, arg) => {
                    v[out] = v[arg].exp();
                }
                RegOp::LnReg(out, arg) => {
                    v[out] = v[arg].ln();
                }
                RegOp::NotReg(out, arg) => {
                    v[out] = v[arg].not();
                }
                RegOp::RandReg(out, _arg) => v[out] = Interval::new(0.0, 1.0),
                RegOp::CopyReg(out, arg) => v[out] = v[arg],
                RegOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm.into();
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    v[out] = v[arg].atan2(imm.into());
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm.atan2(v[arg]);
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].atan2(v[rhs]);
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm.into();
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].and_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].and_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].or_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].or_choice(imm.into());
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].rem_euclid(v[rhs]);
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    v[out] = v[arg].rem_euclid(imm.into());
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm).rem_euclid(v[arg]);
                }
                RegOp::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                RegOp::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                RegOp::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                RegOp::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    v[out] = Interval::compare(v[lhs], v[rhs]);
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = Interval::compare(v[arg], imm);
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = Interval::compare(imm, v[arg]);
                }
                // Mix operations may produce literally anything, unless the
                // interval is a single value
                RegOp::MixRegReg(out, lhs, rhs) => {
                    let lhs = v[lhs];
                    let rhs = v[rhs];
                    // TODO should we do bitwise comparisons here instead of
                    // floating-point comparisons?
                    v[out] = if lhs.lower() == lhs.upper()
                        && rhs.lower() == rhs.upper()
                    {
                        f32::from_bits(crate::rng::mix(
                            lhs.lower().to_bits(),
                            rhs.lower().to_bits(),
                        ))
                        .into()
                    } else {
                        f32::NAN.into()
                    };
                }
                RegOp::MixRegImm(out, arg, imm) => {
                    let arg = v[arg];
                    v[out] = if arg.lower() == arg.upper() {
                        f32::from_bits(crate::rng::mix(
                            arg.lower().to_bits(),
                            imm.to_bits(),
                        ))
                        .into()
                    } else {
                        f32::NAN.into()
                    }
                }
                RegOp::MixImmReg(out, arg, imm) => {
                    let arg = v[arg];
                    v[out] = if arg.lower() == arg.upper() {
                        f32::from_bits(crate::rng::mix(
                            imm.to_bits(),
                            arg.lower().to_bits(),
                        ))
                        .into()
                    } else {
                        f32::NAN.into()
                    }
                }

                RegOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::CopyImm(out, imm) => {
                    v[out] = imm.into();
                }
                RegOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                RegOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        Ok((
            &self.0.out,
            if simplify {
                Some(&self.0.choices)
            } else {
                None
            },
        ))
    }
}

/// VM-based tracing evaluator for single points
#[derive(Default)]
pub struct VmPointEval<const N: usize>(TracingVmEval<f32>);
impl<const N: usize> TracingEvaluator for VmPointEval<N> {
    type Data = f32;
    type Tape = GenericVmTape<N>;
    type Trace = VmTrace;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval(
        &mut self,
        tape: &Self::Tape,
        vars: &[f32],
    ) -> Result<(&[f32], Option<&VmTrace>), TracingEvalError> {
        tape.vars().check_tracing_arguments(vars)?;
        let tape = tape.data();
        self.0.resize_slots(tape);

        let mut choices = self.0.choices.as_mut_slice().iter_mut();
        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize] = v[arg];
                }
                RegOp::Input(out, i) => {
                    v[out] = vars[i as usize];
                }
                RegOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                RegOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RegOp::RecipReg(out, arg) => {
                    v[out] = 1.0 / v[arg];
                }
                RegOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                RegOp::SquareReg(out, arg) => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                RegOp::FloorReg(out, arg) => {
                    v[out] = v[arg].floor();
                }
                RegOp::CeilReg(out, arg) => {
                    v[out] = v[arg].ceil();
                }
                RegOp::RoundReg(out, arg) => {
                    v[out] = v[arg].round();
                }
                RegOp::SinReg(out, arg) => {
                    v[out] = v[arg].sin();
                }
                RegOp::CosReg(out, arg) => {
                    v[out] = v[arg].cos();
                }
                RegOp::TanReg(out, arg) => {
                    v[out] = v[arg].tan();
                }
                RegOp::AsinReg(out, arg) => {
                    v[out] = v[arg].asin();
                }
                RegOp::AcosReg(out, arg) => {
                    v[out] = v[arg].acos();
                }
                RegOp::AtanReg(out, arg) => {
                    v[out] = v[arg].atan();
                }
                RegOp::ExpReg(out, arg) => {
                    v[out] = v[arg].exp();
                }
                RegOp::LnReg(out, arg) => {
                    v[out] = v[arg].ln();
                }
                RegOp::NotReg(out, arg) => v[out] = (v[arg] == 0.0).into(),
                RegOp::RandReg(out, arg) => {
                    v[out] = crate::rng::rand(v[arg].to_bits())
                }
                RegOp::CopyReg(out, arg) => {
                    v[out] = v[arg];
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm;
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm;
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    v[out] = imm / v[arg];
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    v[out] = v[arg].atan2(imm);
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    v[out] = imm.atan2(v[arg]);
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].atan2(v[rhs]);
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].and_choice(imm);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].or_choice(imm);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].rem_euclid(v[rhs]);
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    v[out] = v[arg].rem_euclid(imm);
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    v[out] = imm.rem_euclid(v[arg]);
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] + v[rhs];
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] * v[rhs];
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] / v[rhs];
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs].compare(v[rhs]);
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = v[arg].compare(imm);
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = imm.compare(v[arg]);
                }
                RegOp::MixRegReg(out, lhs, rhs) => {
                    v[out] = f32::from_bits(crate::rng::mix(
                        v[lhs].to_bits(),
                        v[rhs].to_bits(),
                    ))
                }
                RegOp::MixRegImm(out, arg, imm) => {
                    v[out] = f32::from_bits(crate::rng::mix(
                        v[arg].to_bits(),
                        imm.to_bits(),
                    ))
                }
                RegOp::MixImmReg(out, arg, imm) => {
                    v[out] = f32::from_bits(crate::rng::mix(
                        imm.to_bits(),
                        v[arg].to_bits(),
                    ))
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].and_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].or_choice(v[rhs]);
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::CopyImm(out, imm) => {
                    v[out] = imm;
                }
                RegOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                RegOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        Ok((
            &self.0.out,
            if simplify {
                Some(&self.0.choices)
            } else {
                None
            },
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Bulk evaluator for VM tapes
#[derive(Default)]
struct BulkVmEval<T> {
    /// Workspace for data
    slots: Vec<Vec<T>>,

    /// Output array
    out: Vec<Vec<T>>,
}

impl<T: From<f32> + Clone> BulkVmEval<T> {
    /// Reserves slots for the given tape and slice size
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>, size: usize) {
        self.slots
            .resize_with(tape.slot_count(), || vec![f32::NAN.into(); size]);
        for s in self.slots.iter_mut() {
            s.resize(size, f32::NAN.into());
        }

        self.out
            .resize_with(tape.output_count(), || vec![f32::NAN.into(); size]);
        for o in self.out.iter_mut() {
            o.resize(size, f32::NAN.into());
        }
    }
}

/// VM-based bulk evaluator for arrays of points, yielding point values
#[derive(Default)]
pub struct VmFloatSliceEval<const N: usize>(BulkVmEval<f32>);
impl<const N: usize> BulkEvaluator for VmFloatSliceEval<N> {
    type Data = f32;
    type Tape = GenericVmTape<N>;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<'_, f32>, BulkEvalError> {
        tape.vars().check_bulk_arguments(vars)?;
        let tape = tape.data();

        let size = vars.first().map(|v| v.len()).unwrap_or(0);
        self.0.resize_slots(tape, size);

        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize][0..size]
                        .copy_from_slice(&v[arg][0..size]);
                }
                RegOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(&vars[i as usize]);
                }
                RegOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                RegOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RegOp::RecipReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                RegOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                RegOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                RegOp::FloorReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].floor();
                    }
                }
                RegOp::CeilReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ceil();
                    }
                }
                RegOp::RoundReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].round();
                    }
                }
                RegOp::SinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sin();
                    }
                }
                RegOp::CosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].cos();
                    }
                }
                RegOp::TanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].tan();
                    }
                }
                RegOp::AsinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].asin();
                    }
                }
                RegOp::AcosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].acos();
                    }
                }
                RegOp::AtanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan();
                    }
                }
                RegOp::ExpReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].exp();
                    }
                }
                RegOp::LnReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ln();
                    }
                }
                RegOp::NotReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = (v[arg][i] == 0.0).into();
                    }
                }
                RegOp::RandReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = crate::rng::rand(v[arg][i].to_bits())
                    }
                }
                RegOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan2(imm);
                    }
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm.atan2(v[arg][i]);
                    }
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].atan2(v[rhs][i]);
                    }
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm.compare(v[arg][i]);
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].compare(imm);
                    }
                }
                RegOp::MixRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            v[lhs][i].to_bits(),
                            v[rhs][i].to_bits(),
                        ))
                    }
                }
                RegOp::MixRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            v[arg][i].to_bits(),
                            imm.to_bits(),
                        ))
                    }
                }
                RegOp::MixImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            imm.to_bits(),
                            v[arg][i].to_bits(),
                        ))
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min_choice(imm).0;
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max_choice(imm).0;
                    }
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].and_choice(imm).0;
                    }
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].or_choice(imm).0;
                    }
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].rem_euclid(v[rhs][i]);
                    }
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].rem_euclid(imm);
                    }
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm.rem_euclid(v[arg][i]);
                    }
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].compare(v[rhs][i]);
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min_choice(v[rhs][i]).0;
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max_choice(v[rhs][i]).0;
                    }
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].and_choice(v[rhs][i]).0;
                    }
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].or_choice(v[rhs][i]).0;
                    }
                }
                RegOp::CopyImm(out, imm) => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                RegOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                RegOp::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        Ok(BulkOutput::new(&self.0.out, size))
    }
}

/// VM-based bulk evaluator for arrays of points, yielding gradient values
#[derive(Default)]
pub struct VmGradSliceEval<const N: usize>(BulkVmEval<Grad>);
impl<const N: usize> BulkEvaluator for VmGradSliceEval<N> {
    type Data = Grad;
    type Tape = GenericVmTape<N>;
    type TapeStorage = EmptyTapeStorage;

    #[inline]
    fn eval<V: std::ops::Deref<Target = [Self::Data]>>(
        &mut self,
        tape: &Self::Tape,
        vars: &[V],
    ) -> Result<BulkOutput<'_, Grad>, BulkEvalError> {
        tape.vars().check_bulk_arguments(vars)?;
        let tape = tape.data();
        let size = vars.first().map(|v| v.len()).unwrap_or(0);
        self.0.resize_slots(tape, size);

        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Output(arg, i) => {
                    self.0.out[i as usize][0..size]
                        .copy_from_slice(&v[arg][0..size]);
                }
                RegOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(&vars[i as usize]);
                }
                RegOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                RegOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RegOp::RecipReg(out, arg) => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                RegOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                RegOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                RegOp::FloorReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].floor();
                    }
                }
                RegOp::CeilReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ceil();
                    }
                }
                RegOp::RoundReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].round();
                    }
                }
                RegOp::SinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sin();
                    }
                }
                RegOp::CosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].cos();
                    }
                }
                RegOp::TanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].tan();
                    }
                }
                RegOp::AsinReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].asin();
                    }
                }
                RegOp::AcosReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].acos();
                    }
                }
                RegOp::AtanReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan();
                    }
                }
                RegOp::ExpReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].exp();
                    }
                }
                RegOp::LnReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].ln();
                    }
                }
                RegOp::NotReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].not();
                    }
                }
                RegOp::RandReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] =
                            crate::rng::rand(v[arg][i].v.to_bits()).into();
                    }
                }
                RegOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                RegOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                RegOp::AtanRegImm(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = v[arg][i].atan2(imm);
                    }
                }
                RegOp::AtanImmReg(out, arg, imm) => {
                    let imm = Grad::from(imm);
                    for i in 0..size {
                        v[out][i] = imm.atan2(v[arg][i]);
                    }
                }
                RegOp::AtanRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].atan2(v[rhs][i]);
                    }
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
<<<<<<< conflict 1 of 3
%%%%%%% diff from: ospkuuzt 218cdd42 "Add fidget_core::rng module" (parents of rebased revision)
\\\\\\\        to: ospkuuzt 58b1ca21 "Add fidget_core::rng module" (rebase destination)
-                        let p = imm
-                            .partial_cmp(&v[arg][i].v)
-                            .map(|c| c as i8 as f32)
-                            .unwrap_or(f32::NAN);
-                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
+                        v[out][i] = imm.compare(v[arg][i]);
+++++++ mwymyrtv 99d588bb "Working on mix and rand" (rebased revision)
                        v[out][i] = imm
                            .partial_cmp(&v[arg][i].v)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                            .into();
                    }
                }
                RegOp::MixRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            v[lhs][i].v.to_bits(),
                            v[rhs][i].v.to_bits(),
                        ))
                        .into()
                    }
                }
                RegOp::MixRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            v[arg][i].v.to_bits(),
                            imm.to_bits(),
                        ))
                        .into()
                    }
                }
                RegOp::MixImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = f32::from_bits(crate::rng::mix(
                            imm.to_bits(),
                            v[arg][i].v.to_bits(),
                        ))
                        .into()
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    let imm = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].compare(imm);
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm)
                    }
                }
                RegOp::ModRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].rem_euclid(v[rhs][i]);
                    }
                }
                RegOp::ModRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].rem_euclid(imm.into());
                    }
                }
                RegOp::ModImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = Grad::from(imm).rem_euclid(v[arg][i]);
                    }
                }
                RegOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                RegOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                RegOp::AndRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].and(v[rhs][i]);
                    }
                }
                RegOp::AndRegImm(out, arg, imm) => {
                    let imm = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].and(imm);
                    }
                }
                RegOp::OrRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].or(v[rhs][i]);
                    }
                }
                RegOp::OrRegImm(out, arg, imm) => {
                    let imm = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].or(imm);
                    }
                }
                RegOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                RegOp::CompareRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].compare(v[rhs][i]);
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                RegOp::CopyImm(out, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                RegOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                RegOp::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        Ok(BulkOutput::new(&self.0.out, size))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(VmFunction);
    crate::interval_tests!(VmFunction);
    crate::float_slice_tests!(VmFunction);
    crate::point_tests!(VmFunction);
}
