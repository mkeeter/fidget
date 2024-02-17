//! Simple virtual machine for shape evaluation
use crate::{
    compiler::RegOp,
    context::Node,
    eval::{
        types::{Grad, Interval},
        BulkEvaluator, Shape, ShapeVars, Tape, TracingEvaluator,
    },
    Context, Error,
};
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
pub type VmShape = GenericVmShape<{ u8::MAX }>;

impl Tape for VmShape {
    type Storage = ();
    fn recycle(self) -> Self::Storage {
        // nothing to do here
    }
}

/// VM-backed shape with a configurable number of registers
///
/// You are unlikely to use this directly; [`VmShape`] should be used for
/// VM-based evaluation.
#[derive(Clone)]
pub struct GenericVmShape<const N: u8>(Arc<VmData<N>>);

impl<const N: u8> GenericVmShape<N> {
    /// Build a new shape for VM evaluation
    pub fn new(ctx: &Context, node: Node) -> Result<Self, Error> {
        let d = VmData::new(ctx, node)?;
        Ok(Self(Arc::new(d)))
    }
    pub(crate) fn simplify_inner(
        &self,
        choices: &[Choice],
        storage: VmData<N>,
        workspace: &mut VmWorkspace,
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

impl Shape for VmShape {
    type FloatSliceEval = VmFloatSliceEval;
    type Storage = VmData;
    type Workspace = VmWorkspace;

    type TapeStorage = ();

    fn float_slice_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type GradSliceEval = VmGradSliceEval;
    fn grad_slice_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type PointEval = VmPointEval;
    fn point_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type IntervalEval = VmIntervalEval;
    fn interval_tape(&self, _storage: ()) -> Self {
        self.clone()
    }
    type Trace = Vec<Choice>;
    fn simplify(
        &self,
        trace: &Vec<Choice>,
        storage: VmData,
        workspace: &mut Self::Workspace,
    ) -> Result<Self, Error> {
        self.simplify_inner(trace.as_slice(), storage, workspace)
    }

    fn recycle(self) -> Option<Self::Storage> {
        VmShape::recycle(self)
    }

    fn size(&self) -> usize {
        VmShape::size(self)
    }

    fn tile_sizes_3d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }
}

#[cfg(test)]
impl<const N: u8> TryFrom<(&Context, Node)> for GenericVmShape<N> {
    type Error = Error;
    fn try_from(c: (&Context, Node)) -> Result<Self, Error> {
        Self::new(c.0, c.1)
    }
}

impl<const N: u8> ShapeVars for GenericVmShape<N> {
    fn vars(&self) -> Arc<HashMap<String, u32>> {
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
    choices: Vec<Choice>,
}

impl<T> Default for TracingVmEval<T> {
    fn default() -> Self {
        Self {
            slots: vec![],
            choices: vec![],
        }
    }
}

impl<T: From<f32> + Clone> TracingVmEval<T> {
    fn resize_slots(&mut self, tape: &VmData) {
        self.slots.resize(tape.slot_count(), f32::NAN.into());
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.choices.fill(Choice::Unknown);
    }
}

/// VM-based tracing evaluator for intervals
#[derive(Default)]
pub struct VmIntervalEval(TracingVmEval<Interval>);
impl TracingEvaluator for VmIntervalEval {
    type Data = Interval;
    type Tape = VmShape;
    type Trace = Vec<Choice>;
    type TapeStorage = ();

    fn eval<F: Into<Interval>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(Interval, Option<&Vec<Choice>>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let tape = tape.0.as_ref();
        self.check_arguments(vars, tape.var_count())?;
        self.0.resize_slots(tape);
        assert_eq!(vars.len(), tape.var_count());

        let mut simplify = false;
        let mut choice_index = 0;
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
                RegOp::CopyReg(out, arg) => v[out] = v[arg],
                RegOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm.into();
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
                    self.0.choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    self.0.choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                RegOp::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                RegOp::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                RegOp::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    self.0.choices[choice_index] |= choice;
                    simplify |= choice != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    self.0.choices[choice_index] |= choice;
                    simplify |= choice != Choice::Both;
                    choice_index += 1;
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
pub struct VmPointEval(TracingVmEval<f32>);
impl TracingEvaluator for VmPointEval {
    type Data = f32;
    type Tape = VmShape;
    type Trace = Vec<Choice>;
    type TapeStorage = ();

    fn eval<F: Into<f32>>(
        &mut self,
        tape: &Self::Tape,
        x: F,
        y: F,
        z: F,
        vars: &[f32],
    ) -> Result<(f32, Option<&Vec<Choice>>), Error> {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let tape = tape.0.as_ref();
        self.check_arguments(vars, tape.var_count())?;
        self.0.resize_slots(tape);

        let mut choice_index = 0;
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
                    v[out] = if a < imm {
                        self.0.choices[choice_index] |= Choice::Left;
                        a
                    } else if imm < a {
                        self.0.choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        self.0.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= self.0.choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a > imm {
                        self.0.choices[choice_index] |= Choice::Left;
                        a
                    } else if imm > a {
                        self.0.choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        self.0.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= self.0.choices[choice_index] != Choice::Both;
                    choice_index += 1;
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
                RegOp::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    v[out] = if a < b {
                        self.0.choices[choice_index] |= Choice::Left;
                        a
                    } else if b < a {
                        self.0.choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        self.0.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= self.0.choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    v[out] = if a > b {
                        self.0.choices[choice_index] |= Choice::Left;
                        a
                    } else if b > a {
                        self.0.choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        self.0.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= self.0.choices[choice_index] != Choice::Both;
                    choice_index += 1;
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
    fn resize_slots(&mut self, tape: &VmData, size: usize) {
        assert!(tape.reg_limit() == u8::MAX);
        self.slots
            .resize_with(tape.slot_count(), || vec![f32::NAN.into(); size]);
        for s in self.slots.iter_mut() {
            s.resize(size, f32::NAN.into());
        }
    }
}

/// VM-based bulk evaluator for arrays of points, yielding point values
#[derive(Default)]
pub struct VmFloatSliceEval(BulkVmEval<f32>);
impl BulkEvaluator for VmFloatSliceEval {
    type Data = f32;
    type Tape = VmShape;
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
                RegOp::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
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
pub struct VmGradSliceEval(BulkVmEval<Grad>);
impl BulkEvaluator for VmGradSliceEval {
    type Data = Grad;
    type Tape = VmShape;
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
                RegOp::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
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
