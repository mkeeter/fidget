use crate::{
    eval::{
        bulk::{BulkEvaluator, BulkEvaluatorData},
        tracing::{TracingEvaluator, TracingEvaluatorData},
        types::{Grad, Interval},
        Choice, EvaluatorStorage, Family, Tape,
    },
    tape::Op,
};

////////////////////////////////////////////////////////////////////////////////

/// Family of evaluators that use a local interpreter
#[derive(Clone)]
pub enum Eval {}

impl Family for Eval {
    /// This is interpreted, so we can use the maximum number of registers
    const REG_LIMIT: u8 = u8::MAX;

    type IntervalEval = AsmEval;
    type PointEval = AsmEval;
    type FloatSliceEval = AsmEval;
    type GradSliceEval = AsmEval;

    fn tile_sizes_3d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
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

/// Generic tracing evaluator
#[derive(Clone)]
pub struct AsmEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape<Eval>,
}

/// Generic scratch data a tracing evaluator
pub struct AsmTracingEvalData<T> {
    slots: Vec<T>,
}

impl<T> Default for AsmTracingEvalData<T> {
    fn default() -> Self {
        Self { slots: vec![] }
    }
}

impl<T> TracingEvaluatorData<Eval> for AsmTracingEvalData<T>
where
    T: From<f32> + Clone,
{
    fn prepare(&mut self, tape: &Tape<Eval>) {
        assert!(tape.reg_limit() == u8::MAX);

        let slot_count = tape.slot_count();
        self.slots.resize(slot_count, T::from(std::f32::NAN));
        self.slots.fill(T::from(std::f32::NAN));
    }
}

impl EvaluatorStorage<Eval> for AsmEval {
    type Storage = ();
    fn new_with_storage(tape: &Tape<Eval>, _storage: ()) -> Self {
        Self { tape: tape.clone() }
    }
    fn take(self) -> Option<Self::Storage> {
        Some(())
    }
}

////////////////////////////////////////////////////////////////////////////////

impl TracingEvaluator<Interval, Eval> for AsmEval {
    type Data = AsmTracingEvalData<Interval>;

    fn eval_with(
        &self,
        x: Interval,
        y: Interval,
        z: Interval,
        vars: &[f32],
        choices: &mut [Choice],
        data: &mut Self::Data,
    ) -> (Interval, bool) {
        let mut simplify = false;
        assert_eq!(vars.len(), self.tape.var_count());

        let mut choice_index = 0;
        let mut v = SlotArray(&mut data.slots);
        for op in self.tape.iter_asm() {
            match op {
                Op::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                Op::Var(out, i) => {
                    v[out] = vars[i as usize].into();
                }
                Op::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                Op::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                Op::RecipReg(out, arg) => {
                    v[out] = v[arg].recip();
                }
                Op::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                Op::SquareReg(out, arg) => {
                    v[out] = v[arg].square();
                }
                Op::CopyReg(out, arg) => v[out] = v[arg],
                Op::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                Op::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm.into();
                }
                Op::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm.into();
                }
                Op::DivImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                Op::SubImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                Op::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm.into();
                }
                Op::MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                Op::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                Op::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                Op::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                Op::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                Op::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                Op::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    simplify |= choice != Choice::Both;
                    choice_index += 1;
                }
                Op::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    simplify |= choice != Choice::Both;
                    choice_index += 1;
                }
                Op::CopyImm(out, imm) => {
                    v[out] = imm.into();
                }
                Op::Load(out, mem) => {
                    v[out] = v[mem];
                }
                Op::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        (data.slots[0], simplify)
    }
}

impl TracingEvaluator<f32, Eval> for AsmEval {
    type Data = AsmTracingEvalData<f32>;

    fn eval_with(
        &self,
        x: f32,
        y: f32,
        z: f32,
        vars: &[f32],
        choices: &mut [Choice],
        data: &mut Self::Data,
    ) -> (f32, bool) {
        assert_eq!(vars.len(), self.tape.var_count());
        let mut choice_index = 0;
        let mut simplify = false;
        let mut v = SlotArray(&mut data.slots);
        for op in self.tape.iter_asm() {
            match op {
                Op::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                Op::Var(out, i) => v[out] = vars[i as usize],
                Op::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                Op::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                Op::RecipReg(out, arg) => {
                    v[out] = 1.0 / v[arg];
                }
                Op::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                Op::SquareReg(out, arg) => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                Op::CopyReg(out, arg) => {
                    v[out] = v[arg];
                }
                Op::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm;
                }
                Op::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                Op::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm;
                }
                Op::DivImmReg(out, arg, imm) => {
                    v[out] = imm / v[arg];
                }
                Op::SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                Op::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                Op::MinRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a < imm {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if imm < a {
                        choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            std::f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                Op::MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a > imm {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if imm > a {
                        choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        choices[choice_index] |= Choice::Both;
                        if a.is_nan() || imm.is_nan() {
                            std::f32::NAN
                        } else {
                            imm
                        }
                    };
                    simplify |= choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                Op::AddRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] + v[rhs];
                }
                Op::MulRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] * v[rhs];
                }
                Op::DivRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] / v[rhs];
                }
                Op::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                Op::MinRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    v[out] = if a < b {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if b < a {
                        choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            std::f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                Op::MaxRegReg(out, lhs, rhs) => {
                    let a = v[lhs];
                    let b = v[rhs];
                    v[out] = if a > b {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if b > a {
                        choices[choice_index] |= Choice::Right;
                        b
                    } else {
                        choices[choice_index] |= Choice::Both;
                        if a.is_nan() || b.is_nan() {
                            std::f32::NAN
                        } else {
                            b
                        }
                    };
                    simplify |= choices[choice_index] != Choice::Both;
                    choice_index += 1;
                }
                Op::CopyImm(out, imm) => {
                    v[out] = imm;
                }
                Op::Load(out, mem) => {
                    v[out] = v[mem];
                }
                Op::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        (data.slots[0], simplify)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`Op`]
pub struct AsmBulkEvalData<T> {
    /// Workspace for data
    slots: Vec<Vec<T>>,
    /// Current slice size in `self.slots`
    slice_size: usize,
}

impl<T> Default for AsmBulkEvalData<T> {
    fn default() -> Self {
        Self {
            slots: vec![],
            slice_size: 0,
        }
    }
}

impl<T> BulkEvaluatorData<Eval> for AsmBulkEvalData<T>
where
    T: From<f32> + Clone,
{
    fn prepare(&mut self, tape: &Tape<Eval>, size: usize) {
        assert!(tape.reg_limit() == u8::MAX);
        self.slots.resize_with(tape.slot_count(), || {
            vec![std::f32::NAN.into(); size.max(self.slice_size)]
        });
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, std::f32::NAN.into());
            }
            self.slice_size = size;
        }
    }
}

impl BulkEvaluator<f32, Eval> for AsmEval {
    type Data = AsmBulkEvalData<f32>;

    fn eval_with(
        &self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [f32],
        data: &mut Self::Data,
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.tape.var_count());
        assert_eq!(data.slots.len(), self.tape.slot_count());

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.tape.iter_asm() {
            match op {
                Op::Input(out, i) => v[out][0..size].copy_from_slice(match i {
                    0 => xs,
                    1 => ys,
                    2 => zs,
                    _ => panic!("Invalid input: {}", i),
                }),
                Op::Var(out, i) => v[out][0..size].fill(vars[i as usize]),
                Op::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                Op::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                Op::RecipReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                Op::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                Op::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                Op::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                Op::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                Op::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                Op::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                Op::DivImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                Op::SubImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                Op::SubRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                Op::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                Op::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                Op::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                Op::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                Op::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                Op::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                Op::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                Op::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                Op::CopyImm(out, imm) => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Op::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                Op::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        out[0..size].copy_from_slice(&data.slots[0][0..size])
    }
}

////////////////////////////////////////////////////////////////////////////////

impl BulkEvaluator<Grad, Eval> for AsmEval {
    type Data = AsmBulkEvalData<Grad>;

    fn eval_with(
        &self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [Grad],
        data: &mut Self::Data,
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.tape.var_count());
        assert_eq!(data.slots.len(), self.tape.slot_count());

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.tape.iter_asm() {
            match op {
                Op::Input(out, j) => {
                    for i in 0..size {
                        v[out][i] = match j {
                            0 => Grad::new(xs[i], 1.0, 0.0, 0.0),
                            1 => Grad::new(ys[i], 0.0, 1.0, 0.0),
                            2 => Grad::new(zs[i], 0.0, 0.0, 1.0),
                            _ => panic!("Invalid input: {}", i),
                        }
                    }
                }
                Op::Var(out, j) => {
                    // TODO: error handling?
                    v[out][0..size].fill(Grad::new(
                        vars[j as usize],
                        0.0,
                        0.0,
                        0.0,
                    ));
                }
                Op::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                Op::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                Op::RecipReg(out, arg) => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                Op::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                Op::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                Op::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                Op::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                Op::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm.into();
                    }
                }
                Op::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                Op::DivImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                Op::SubImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                Op::SubRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                Op::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                Op::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                Op::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                Op::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                Op::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                Op::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                Op::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                Op::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                Op::CopyImm(out, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Op::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                Op::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        out.copy_from_slice(&data.slots[0][0..size])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(Eval);
    crate::interval_tests!(Eval);
    crate::float_slice_tests!(Eval);
    crate::point_tests!(Eval);
}
