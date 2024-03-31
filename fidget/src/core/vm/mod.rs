//! Simple virtual machine for shape evaluation
use crate::{
    compiler::RegOp,
    context::Node,
    eval::{
        BulkEvaluator, MathShape, Shape, ShapeVars, Tape, Trace,
        TracingEvaluator, TransformedShape,
    },
    types::{Grad, Interval},
    Context, Error,
};
use nalgebra::Matrix4;
use std::{collections::HashMap, sync::Arc};

mod choice;
mod data;

pub use choice::Choice;
pub use data::{VmData, VmWorkspace};

////////////////////////////////////////////////////////////////////////////////

/// Shape that use a VM backend for evaluation
///
/// Internally, the [`VmShape`] stores an [`Arc<VmData>`](VmData), and
/// iterates over a [`Vec<RegOp>`](RegOp) to perform evaluation.
///
/// All of the associated [`Tape`] types simply clone the internal `Arc`;
/// there's no separate planning required to generate a tape.
///
pub type VmShape = GenericVmShape<{ u8::MAX as usize }>;

impl<const N: usize> Tape for GenericVmShape<N> {
    type Storage = ();
    fn recycle(self) -> Self::Storage {
        // nothing to do here
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

#[cfg(test)]
impl From<Vec<Choice>> for VmTrace {
    fn from(v: Vec<Choice>) -> Self {
        Self(v)
    }
}

#[cfg(test)]
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
pub struct GenericVmShape<const N: usize>(Arc<VmData<N>>);

impl<const N: usize> GenericVmShape<N> {
    pub(crate) fn simplify_inner(
        &self,
        choices: &[Choice],
        storage: VmData<N>,
        workspace: &mut VmWorkspace<N>,
    ) -> Result<Self, Error> {
        let d = self.0.simplify(choices, workspace, storage)?;
        Ok(Self(Arc::new(d)))
    }
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

    /// Returns the number of variables in the tape
    pub fn var_count(&self) -> usize {
        self.0.var_count()
    }

    /// Returns the number of choices (i.e. `min` and `max` nodes) in the tape
    pub fn choice_count(&self) -> usize {
        self.0.choice_count()
    }
}

impl<const N: usize> Shape for GenericVmShape<N> {
    type FloatSliceEval = VmFloatSliceEval<N>;
    type Storage = VmData<N>;
    type Workspace = VmWorkspace<N>;

    type TapeStorage = ();

    fn float_slice_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type GradSliceEval = VmGradSliceEval<N>;
    fn grad_slice_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type PointEval = VmPointEval<N>;
    fn point_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type IntervalEval = VmIntervalEval<N>;
    fn interval_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type Trace = VmTrace;
    fn simplify(
        &self,
        trace: &VmTrace,
        storage: VmData<N>,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error> {
        self.simplify_inner(trace.as_slice(), storage, workspace)
    }

    fn recycle(self) -> Option<Self::Storage> {
        GenericVmShape::recycle(self)
    }

    fn size(&self) -> usize {
        GenericVmShape::size(self)
    }

    fn tile_sizes_3d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    type TransformedShape = TransformedShape<Self>;
    fn apply_transform(self, mat: Matrix4<f32>) -> Self::TransformedShape {
        TransformedShape::new(self, mat)
    }
}

impl<const N: usize> MathShape for GenericVmShape<N> {
    fn new(ctx: &Context, node: Node) -> Result<Self, Error> {
        let d = VmData::new(ctx, node)?;
        Ok(Self(Arc::new(d)))
    }
}

impl<const N: usize> ShapeVars for GenericVmShape<N> {
    fn vars(&self) -> &HashMap<String, u32> {
        self.0.vars()
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
    choices: VmTrace,
}

impl<T> Default for TracingVmEval<T> {
    fn default() -> Self {
        Self {
            slots: vec![],
            choices: VmTrace::default(),
        }
    }
}

impl<T: From<f32> + Clone> TracingVmEval<T> {
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>) {
        self.slots.resize(tape.slot_count(), f32::NAN.into());
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.choices.fill(Choice::Unknown);
    }
}

/// VM-based tracing evaluator for intervals
#[derive(Default)]
pub struct VmIntervalEval<const N: usize>(TracingVmEval<Interval>);
impl<const N: usize> TracingEvaluator for VmIntervalEval<N> {
    type Data = Interval;
    type Tape = GenericVmShape<N>;
    type Trace = VmTrace;
    type TapeStorage = ();

    fn eval<F: Into<Interval>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(Interval, Option<&VmTrace>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let tape = tape.0.as_ref();
        self.check_arguments(vars, tape.var_count())?;
        self.0.resize_slots(tape);
        assert_eq!(vars.len(), tape.var_count());

        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        let mut choices = self.0.choices.as_mut_slice().iter_mut();
        for op in tape.iter_asm() {
            match op {
                RegOp::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                RegOp::Var(out, i) => {
                    v[out] = vars[i as usize].into();
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
                    v[out] = if v[lhs].has_nan() || v[rhs].has_nan() {
                        f32::NAN.into()
                    } else if v[lhs].upper() < v[rhs].lower() {
                        Interval::from(-1.0)
                    } else if v[lhs].lower() > v[rhs].upper() {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = if v[arg].has_nan() || imm.is_nan() {
                        f32::NAN.into()
                    } else if v[arg].upper() < imm {
                        Interval::from(-1.0)
                    } else if v[arg].lower() > imm {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = if v[arg].has_nan() || imm.is_nan() {
                        f32::NAN.into()
                    } else if imm < v[arg].lower() {
                        Interval::from(-1.0)
                    } else if imm > v[arg].upper() {
                        Interval::from(1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
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
            self.0.slots[0],
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
    type Tape = GenericVmShape<N>;
    type Trace = VmTrace;
    type TapeStorage = ();

    fn eval<F: Into<f32>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(f32, Option<&VmTrace>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let tape = tape.0.as_ref();
        self.check_arguments(vars, tape.var_count())?;
        self.0.resize_slots(tape);

        let mut choices = self.0.choices.as_mut_slice().iter_mut();
        let mut simplify = false;
        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                RegOp::Var(out, i) => v[out] = vars[i as usize],
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
                RegOp::SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a < imm {
                        (Choice::Left, a)
                    } else if imm < a {
                        (Choice::Right, imm)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || imm.is_nan() {
                                f32::NAN
                            } else {
                                imm
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    let (choice, value) = if a > imm {
                        (Choice::Left, a)
                    } else if imm > a {
                        (Choice::Right, imm)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || imm.is_nan() {
                                f32::NAN
                            } else {
                                imm
                            },
                        )
                    };
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
                    v[out] = v[lhs]
                        .partial_cmp(&v[rhs])
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    v[out] = v[arg]
                        .partial_cmp(&imm)
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::CompareImmReg(out, arg, imm) => {
                    v[out] = imm
                        .partial_cmp(&v[arg])
                        .map(|c| c as i8 as f32)
                        .unwrap_or(f32::NAN)
                }
                RegOp::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a < b {
                        (Choice::Left, a)
                    } else if b < a {
                        (Choice::Right, b)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || b.is_nan() {
                                f32::NAN
                            } else {
                                b
                            },
                        )
                    };
                    v[out] = value;
                    *choices.next().unwrap() |= choice;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    let (choice, value) = if a > b {
                        (Choice::Left, a)
                    } else if b > a {
                        (Choice::Right, b)
                    } else {
                        (
                            Choice::Both,
                            if a.is_nan() || b.is_nan() {
                                f32::NAN
                            } else {
                                b
                            },
                        )
                    };
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
            self.0.slots[0],
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
}

impl<T: From<f32> + Clone> BulkVmEval<T> {
    /// Reserves slots for the given tape and slice size
    fn resize_slots<const N: usize>(&mut self, tape: &VmData<N>, size: usize) {
        self.slots
            .resize_with(tape.slot_count(), || vec![f32::NAN.into(); size]);
        for s in self.slots.iter_mut() {
            s.resize(size, f32::NAN.into());
        }
    }
}

/// VM-based bulk evaluator for arrays of points, yielding point values
#[derive(Default)]
pub struct VmFloatSliceEval<const N: usize>(BulkVmEval<f32>);
impl<const N: usize> BulkEvaluator for VmFloatSliceEval<N> {
    type Data = f32;
    type Tape = GenericVmShape<N>;
    type TapeStorage = ();

    fn eval(
        &mut self,
        tape: &Self::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<&[f32], Error> {
        let tape = tape.0.as_ref();
        self.check_arguments(xs, ys, zs, vars, tape.var_count())?;
        self.0.resize_slots(tape, xs.len());
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(vars.len(), tape.var_count());

        let size = xs.len();

        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(match i {
                        0 => xs,
                        1 => ys,
                        2 => zs,
                        _ => panic!("Invalid input: {}", i),
                    })
                }
                RegOp::Var(out, i) => v[out][0..size].fill(vars[i as usize]),
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
                        v[out][i] = imm
                            .partial_cmp(&v[arg][i])
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i]
                            .partial_cmp(&imm)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            v[arg][i].min(imm)
                        };
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = if v[arg][i].is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            v[arg][i].max(imm)
                        };
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
                        v[out][i] = v[lhs][i]
                            .partial_cmp(&v[rhs][i])
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN)
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].is_nan() || v[rhs][i].is_nan()
                        {
                            f32::NAN
                        } else {
                            v[lhs][i].min(v[rhs][i])
                        };
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = if v[lhs][i].is_nan() || v[rhs][i].is_nan()
                        {
                            f32::NAN
                        } else {
                            v[lhs][i].max(v[rhs][i])
                        };
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
        Ok(&self.0.slots[0])
    }
}

/// VM-based bulk evaluator for arrays of points, yielding gradient values
#[derive(Default)]
pub struct VmGradSliceEval<const N: usize>(BulkVmEval<Grad>);
impl<const N: usize> BulkEvaluator for VmGradSliceEval<N> {
    type Data = Grad;
    type Tape = GenericVmShape<N>;
    type TapeStorage = ();

    fn eval(
        &mut self,
        tape: &Self::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<&[Grad], Error> {
        let tape = tape.0.as_ref();
        self.check_arguments(xs, ys, zs, vars, tape.var_count())?;
        self.0.resize_slots(tape, xs.len());
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(vars.len(), tape.var_count());

        let size = xs.len();
        let mut v = SlotArray(&mut self.0.slots);
        for op in tape.iter_asm() {
            match op {
                RegOp::Input(out, j) => {
                    for i in 0..size {
                        v[out][i] = match j {
                            0 => Grad::new(xs[i], 1.0, 0.0, 0.0),
                            1 => Grad::new(ys[i], 0.0, 1.0, 0.0),
                            2 => Grad::new(zs[i], 0.0, 0.0, 1.0),
                            _ => panic!("Invalid input: {}", i),
                        }
                    }
                }
                RegOp::Var(out, j) => {
                    // TODO: error handling?
                    v[out][0..size].fill(Grad::new(
                        vars[j as usize],
                        0.0,
                        0.0,
                        0.0,
                    ));
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
                        v[out][i] = v[arg][i] * imm.into();
                    }
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
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
                    for i in 0..size {
                        let p = imm
                            .partial_cmp(&v[arg][i].v)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::CompareRegImm(out, arg, imm) => {
                    for i in 0..size {
                        let p = v[arg][i]
                            .v
                            .partial_cmp(&imm)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v.is_nan() || imm.v.is_nan() {
                            f32::NAN.into()
                        } else {
                            v[arg][i].min(imm)
                        };
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = if v[arg][i].v.is_nan() || imm.v.is_nan() {
                            f32::NAN.into()
                        } else {
                            v[arg][i].max(imm)
                        };
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
                        let p = v[lhs][i]
                            .v
                            .partial_cmp(&v[rhs][i].v)
                            .map(|c| c as i8 as f32)
                            .unwrap_or(f32::NAN);
                        v[out][i] = Grad::new(p, 0.0, 0.0, 0.0);
                    }
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[lhs][i].v.is_nan() || v[rhs][i].v.is_nan() {
                                f32::NAN.into()
                            } else {
                                v[lhs][i].min(v[rhs][i])
                            };
                    }
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] =
                            if v[lhs][i].v.is_nan() || v[rhs][i].v.is_nan() {
                                f32::NAN.into()
                            } else {
                                v[lhs][i].max(v[rhs][i])
                            };
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
        Ok(&self.0.slots[0])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(VmShape);
    crate::interval_tests!(VmShape);
    crate::float_slice_tests!(VmShape);
    crate::point_tests!(VmShape);
}
