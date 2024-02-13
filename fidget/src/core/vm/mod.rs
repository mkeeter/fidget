//! Shapes that use a VM backend for evaluation
use crate::{
    compiler::RegOp,
    eval::{
        bulk::BulkEvaluator,
        tracing::TracingEvaluator,
        types::{Grad, Interval},
        Choice, Shape, ShapeVars, TapeData,
    },
    Error,
};
use std::{collections::HashMap, sync::Arc};

////////////////////////////////////////////////////////////////////////////////

/// Shape that use a VM backend for evaluation
#[derive(Clone)]
pub struct VmShape(Arc<TapeData>);

impl VmShape {
    /// Build a new shape for VM evaluation
    pub fn new(
        ctx: &crate::Context,
        node: crate::context::Node,
    ) -> Result<Self, Error> {
        let d = TapeData::new(ctx, node)?;
        Ok(Self(Arc::new(d)))
    }
    fn simplify_inner(&self, choices: &[Choice]) -> Result<Self, Error> {
        let mut workspace = crate::eval::tape::Workspace::default();
        let next = TapeData::default();
        let d = self.0.simplify_with(choices, &mut workspace, next)?;
        Ok(Self(Arc::new(d)))
    }
    /// Returns a characteristic size (the length of the inner assembly tape)
    pub fn size(&self) -> usize {
        self.0.len()
    }
}

impl Shape for VmShape {
    type FloatSliceEval = BulkVmEval<f32>;
    fn float_slice_tape(&self) -> Self {
        self.clone()
    }
    type GradSliceEval = BulkVmEval<Grad>;
    fn grad_slice_tape(&self) -> Self {
        self.clone()
    }
    type PointEval = TracingVmEval<f32>;
    fn point_tape(&self) -> Self {
        self.clone()
    }
    type IntervalEval = TracingVmEval<Interval>;
    fn interval_tape(&self) -> Self {
        self.clone()
    }
    type Trace = Vec<Choice>;
    fn simplify(&self, trace: &Vec<Choice>) -> Result<Self, Error> {
        self.simplify_inner(trace.as_slice())
    }
    #[cfg(test)]
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
impl TryFrom<(&crate::Context, crate::context::Node)> for VmShape {
    type Error = Error;
    fn try_from(
        c: (&crate::Context, crate::context::Node),
    ) -> Result<Self, Error> {
        Self::new(c.0, c.1)
    }
}

impl ShapeVars for VmShape {
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
pub struct TracingVmEval<T> {
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
    fn resize_slots(&mut self, tape: &TapeData) {
        self.slots.resize(tape.slot_count(), f32::NAN.into());
        self.choices.resize(tape.choice_count(), Choice::Unknown);
        self.choices.fill(Choice::Unknown);
    }

    fn check_arguments(
        &self,
        tape: &TapeData,
        vars: &[f32],
    ) -> Result<(), Error> {
        if vars.len() != tape.var_count() {
            Err(Error::BadVarSlice(vars.len(), tape.var_count()))
        } else {
            Ok(())
        }
    }
}

impl TracingEvaluator<Interval, Vec<Choice>> for TracingVmEval<Interval> {
    type Tape = VmShape;

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
        self.check_arguments(tape, vars)?;
        self.resize_slots(tape);
        assert_eq!(vars.len(), tape.var_count());

        let mut simplify = false;
        let mut choice_index = 0;
        let mut v = SlotArray(&mut self.slots);
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
                    self.choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    self.choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                RegOp::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                RegOp::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                RegOp::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                RegOp::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                RegOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    println!("{value} {choice:?}");
                    v[out] = value;
                    self.choices[choice_index] |= choice;
                    simplify |= choice != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    self.choices[choice_index] |= choice;
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
            self.slots[0],
            if simplify { Some(&self.choices) } else { None },
        ))
    }
}

impl TracingEvaluator<f32, Vec<Choice>> for TracingVmEval<f32> {
    type Tape = VmShape;

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
        self.check_arguments(tape, vars)?;
        self.resize_slots(tape);
        assert_eq!(vars.len(), tape.var_count());

        let mut choice_index = 0;
        let mut simplify = false;
        let mut v = SlotArray(&mut self.slots);
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
                        self.choices[choice_index] |= Choice::Left;
                        a
                    } else if imm < a {
                        self.choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        self.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= self.choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a > imm {
                        self.choices[choice_index] |= Choice::Left;
                        a
                    } else if imm > a {
                        self.choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        self.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= self.choices[choice_index] != Choice::Both;
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
                        self.choices[choice_index] |= Choice::Left;
                        a
                    } else if b < a {
                        self.choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        self.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= self.choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                RegOp::MaxRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    v[out] = if a > b {
                        self.choices[choice_index] |= Choice::Left;
                        a
                    } else if b > a {
                        self.choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        self.choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= self.choices[choice_index] != Choice::Both;
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
            self.slots[0],
            if simplify { Some(&self.choices) } else { None },
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`RegOp`]
#[derive(Default)]
pub struct BulkVmEval<T> {
    /// Workspace for data
    slots: Vec<Vec<T>>,
}

impl<T: From<f32> + Clone> BulkVmEval<T> {
    /// Reserves slots for the given tape and slice size
    fn resize_slots(&mut self, tape: &TapeData, size: usize) {
        println!("resizing to {size}");
        assert!(tape.reg_limit() == u8::MAX);
        self.slots
            .resize_with(tape.slot_count(), || vec![f32::NAN.into(); size]);
        for s in self.slots.iter_mut() {
            s.resize(size, f32::NAN.into());
        }
    }

    fn check_arguments(
        &self,
        tape: &TapeData,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<(), Error> {
        if xs.len() != ys.len() || ys.len() != zs.len() {
            Err(Error::MismatchedSlices)
        } else if vars.len() != tape.var_count() {
            Err(Error::BadVarSlice(vars.len(), tape.var_count()))
        } else {
            Ok(())
        }
    }
}

impl BulkEvaluator<f32> for BulkVmEval<f32> {
    type Tape = VmShape;

    fn eval(
        &mut self,
        tape: &Self::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<&[f32], Error> {
        let tape = tape.0.as_ref();
        self.check_arguments(tape, xs, ys, zs, vars)?;
        self.resize_slots(tape, xs.len());
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(vars.len(), tape.var_count());

        let size = xs.len();

        let mut v = SlotArray(&mut self.slots);
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
        Ok(&self.slots[0])
    }
}

////////////////////////////////////////////////////////////////////////////////

impl BulkEvaluator<Grad> for BulkVmEval<Grad> {
    type Tape = VmShape;

    fn eval(
        &mut self,
        tape: &Self::Tape,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
    ) -> Result<&[Grad], Error> {
        let tape = tape.0.as_ref();
        self.check_arguments(tape, xs, ys, zs, vars)?;
        self.resize_slots(tape, xs.len());
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(vars.len(), tape.var_count());

        let size = xs.len();
        let mut v = SlotArray(&mut self.slots);
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
        Ok(&self.slots[0])
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
