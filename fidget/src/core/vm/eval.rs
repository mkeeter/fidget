use crate::{
    eval::{
        float_slice::FloatSliceEvalT,
        grad::{Grad, GradEvalT},
        interval::Interval,
        point::PointEvalT,
        Choice, Family, Tape,
    },
    vm::Op,
};

////////////////////////////////////////////////////////////////////////////////

/// Family of evaluators that use a local interpreter
#[derive(Clone)]
pub enum Eval {}

impl Family for Eval {
    /// This is interpreted, so we can use the maximum number of registers
    const REG_LIMIT: u8 = u8::MAX;

    type IntervalEval = AsmIntervalEval;
    type FloatSliceEval = AsmFloatSliceEval;
    type PointEval = AsmPointEval;
    type GradEval = AsmGradEval;

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

/// Interval evaluator for a slice of [`Op`]
#[derive(Clone)]
pub struct AsmIntervalEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape<Eval>,
}

#[derive(Default)]
pub struct AsmIntervalEvalData {
    /// Workspace for data
    slots: Vec<Interval>,
}

impl crate::eval::TracingEvaluatorData<Eval> for AsmIntervalEvalData {
    fn prepare(&mut self, tape: &Tape<Eval>) {
        assert!(tape.reg_limit() == u8::MAX);

        let slot_count = tape.slot_count();
        self.slots.resize(slot_count, Interval::from(std::f32::NAN));
        self.slots.fill(Interval::from(std::f32::NAN));
    }
}

impl crate::eval::EvaluatorStorage<Eval> for AsmIntervalEval {
    type Storage = ();
    fn new_with_storage(tape: &Tape<Eval>, _storage: ()) -> Self {
        Self { tape: tape.clone() }
    }
    fn take(self) -> Option<Self::Storage> {
        Some(())
    }
}

impl crate::eval::TracingEvaluator<Interval, Eval> for AsmIntervalEval {
    type Data = AsmIntervalEvalData;

    fn eval_with(
        &self,
        x: Interval,
        y: Interval,
        z: Interval,
        vars: &[f32],
        choices: &mut [Choice],
        data: &mut AsmIntervalEvalData,
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
                    choice_index += 1;
                }
                Op::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
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

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`Op`]
pub struct AsmFloatSliceEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape<Eval>,
    /// Workspace for data
    slots: Vec<Vec<f32>>,
    /// Current slice size in `self.slots`
    slice_size: usize,
}

impl FloatSliceEvalT<Eval> for AsmFloatSliceEval {
    type Storage = ();

    fn new(tape: &Tape<Eval>) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape: tape.clone(),
            slots: vec![vec![]; slot_count],
            slice_size: 0,
        }
    }

    fn eval_s(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [f32],
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.tape.var_count());

        let size = xs.len();
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, std::f32::NAN);
            }
            self.slice_size = size;
        }

        let mut v = SlotArray(&mut self.slots);
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
        out[0..size].copy_from_slice(&self.slots[0][0..size])
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`Op`]
pub struct AsmPointEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape<Eval>,
    /// Workspace for data
    slots: Vec<f32>,
}

impl PointEvalT<Eval> for AsmPointEval {
    fn new(tape: &Tape<Eval>) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape: tape.clone(),
            slots: vec![std::f32::NAN; slot_count],
        }
    }
    fn eval_p(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        vars: &[f32],
        choices: &mut [Choice],
    ) -> f32 {
        assert_eq!(vars.len(), self.tape.var_count());
        let mut choice_index = 0;
        let mut v = SlotArray(&mut self.slots);
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
        self.slots[0]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`Op`]
pub struct AsmGradEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape<Eval>,
    /// Workspace for data
    slots: Vec<Vec<Grad>>,
    slice_size: usize,
}

impl GradEvalT<Eval> for AsmGradEval {
    type Storage = ();

    fn new(tape: &Tape<Eval>) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape: tape.clone(),
            slots: vec![vec![]; slot_count],
            slice_size: 0,
        }
    }

    fn eval_f(&mut self, x: f32, y: f32, z: f32, vars: &[f32]) -> Grad {
        let mut out = [Grad::default()];
        self.eval_g(&[x], &[y], &[z], vars, out.as_mut_slice());
        out[0]
    }

    fn eval_g(
        &mut self,
        xs: &[f32],
        ys: &[f32],
        zs: &[f32],
        vars: &[f32],
        out: &mut [Grad],
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.tape.var_count());

        let size = xs.len();
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, Grad::default());
            }
            self.slice_size = size;
        }
        let mut v = SlotArray(&mut self.slots);
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
        out.copy_from_slice(&self.slots[0][0..size])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_tests!(Eval);
    crate::interval_tests!(Eval);
    crate::float_slice_tests!(Eval);
    crate::point_tests!(Eval);
}
