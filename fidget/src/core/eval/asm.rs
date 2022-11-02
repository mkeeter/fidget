use crate::{
    asm::AsmOp,
    eval::{
        float_slice::FloatSliceEvalT,
        grad::{Grad, GradEvalT},
        interval::{Interval, IntervalEvalT},
        point::PointEvalT,
        Choice, EvalFamily,
    },
    tape::Tape,
};

////////////////////////////////////////////////////////////////////////////////

/// Family of evaluators that use a local interpreter
pub enum AsmFamily {}

impl EvalFamily for AsmFamily {
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

/// Interval evaluator for a slice of [`AsmOp`]
#[derive(Clone)]
pub struct AsmIntervalEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Interval>,
}

impl From<Tape> for AsmIntervalEval {
    fn from(tape: Tape) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape,
            slots: vec![Interval::from(std::f32::NAN); slot_count],
        }
    }
}

impl IntervalEvalT for AsmIntervalEval {
    type Storage = ();
    type Family = AsmFamily;

    fn take(self) -> Option<Self::Storage> {
        Some(())
    }

    fn from_tape_give(tape: Tape, _prev: Self::Storage) -> Self {
        tape.into()
    }

    fn eval_i<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
        choices: &mut [Choice],
    ) -> Interval {
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let mut choice_index = 0;
        let mut v = SlotArray(&mut self.slots);
        for op in self.tape.iter_asm() {
            match op {
                AsmOp::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                AsmOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                AsmOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                AsmOp::RecipReg(out, arg) => {
                    v[out] = v[arg].recip();
                }
                AsmOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                AsmOp::SquareReg(out, arg) => {
                    v[out] = v[arg].square();
                }
                AsmOp::CopyReg(out, arg) => v[out] = v[arg],
                AsmOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                AsmOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm.into();
                }
                AsmOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm.into();
                }
                AsmOp::DivImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                AsmOp::SubImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                AsmOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm.into();
                }
                AsmOp::MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AsmOp::MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AsmOp::AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                AsmOp::MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                AsmOp::DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                AsmOp::SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                AsmOp::MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AsmOp::MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AsmOp::CopyImm(out, imm) => {
                    v[out] = imm.into();
                }
                AsmOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                AsmOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        self.slots[0]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmFloatSliceEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Vec<f32>>,
    /// Current slice size in `self.slots`
    slice_size: usize,
}

impl From<Tape> for AsmFloatSliceEval {
    fn from(tape: Tape) -> Self {
        Self::new(tape)
    }
}

impl FloatSliceEvalT for AsmFloatSliceEval {
    type Storage = ();
    type Family = AsmFamily;

    fn from_tape_give(tape: Tape, _s: Self::Storage) -> Self {
        Self::from(tape)
    }

    fn take(self) -> Option<()> {
        // There is no storage, so you can always take it
        Some(())
    }

    fn eval_s(&mut self, xs: &[f32], ys: &[f32], zs: &[f32], out: &mut [f32]) {
        let size = [xs.len(), ys.len(), zs.len(), out.len()]
            .into_iter()
            .min()
            .unwrap();
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, std::f32::NAN);
            }
            self.slice_size = size;
        }

        let mut v = SlotArray(&mut self.slots);
        for op in self.tape.iter_asm() {
            match op {
                AsmOp::Input(out, i) => {
                    v[out][0..size].copy_from_slice(match i {
                        0 => xs,
                        1 => ys,
                        2 => zs,
                        _ => panic!("Invalid input: {}", i),
                    })
                }
                AsmOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                AsmOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                AsmOp::RecipReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                AsmOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                AsmOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                AsmOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                AsmOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                AsmOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                AsmOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                AsmOp::DivImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                AsmOp::SubImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                AsmOp::SubRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                AsmOp::MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                AsmOp::MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                AsmOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                AsmOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                AsmOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                AsmOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                AsmOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                AsmOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                AsmOp::CopyImm(out, imm) => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                AsmOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                AsmOp::Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        out[0..size].copy_from_slice(&self.slots[0][0..size])
    }
}

impl AsmFloatSliceEval {
    pub fn new(tape: Tape) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape,
            slots: vec![vec![]; slot_count],
            slice_size: 0,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmPointEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<f32>,
}

impl From<Tape> for AsmPointEval {
    fn from(tape: Tape) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape,
            slots: vec![std::f32::NAN; slot_count],
        }
    }
}

impl PointEvalT for AsmPointEval {
    type Family = AsmFamily;
    fn eval_p(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        choices: &mut [Choice],
    ) -> f32 {
        let mut choice_index = 0;
        let mut v = SlotArray(&mut self.slots);
        for op in self.tape.iter_asm() {
            match op {
                AsmOp::Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                AsmOp::NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                AsmOp::AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                AsmOp::RecipReg(out, arg) => {
                    v[out] = 1.0 / v[arg];
                }
                AsmOp::SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                AsmOp::SquareReg(out, arg) => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                AsmOp::CopyReg(out, arg) => {
                    v[out] = v[arg];
                }
                AsmOp::AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm;
                }
                AsmOp::MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                AsmOp::DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm;
                }
                AsmOp::DivImmReg(out, arg, imm) => {
                    v[out] = imm / v[arg];
                }
                AsmOp::SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                AsmOp::SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                AsmOp::MinRegImm(out, arg, imm) => {
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
                AsmOp::MaxRegImm(out, arg, imm) => {
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
                AsmOp::AddRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] + v[rhs];
                }
                AsmOp::MulRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] * v[rhs];
                }
                AsmOp::DivRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] / v[rhs];
                }
                AsmOp::SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                AsmOp::MinRegReg(out, lhs, rhs) => {
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
                AsmOp::MaxRegReg(out, lhs, rhs) => {
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
                AsmOp::CopyImm(out, imm) => {
                    v[out] = imm;
                }
                AsmOp::Load(out, mem) => {
                    v[out] = v[mem];
                }
                AsmOp::Store(out, mem) => {
                    v[mem] = v[out];
                }
            }
        }
        self.slots[0]
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmGradEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Vec<Grad>>,
    slice_size: usize,
}

impl From<Tape> for AsmGradEval {
    fn from(tape: Tape) -> Self {
        let slot_count = tape.slot_count();
        Self {
            tape,
            slots: vec![vec![]; slot_count],
            slice_size: 0,
        }
    }
}

impl GradEvalT for AsmGradEval {
    type Storage = ();
    type Family = AsmFamily;

    fn take(self) -> Option<Self::Storage> {
        Some(())
    }

    fn from_tape_give(tape: Tape, _prev: Self::Storage) -> Self {
        tape.into()
    }

    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> Grad {
        let mut out = [Grad::default()];
        self.eval_g(&[x], &[y], &[z], out.as_mut_slice());
        out[0]
    }

    fn eval_g(&mut self, xs: &[f32], ys: &[f32], zs: &[f32], out: &mut [Grad]) {
        let size = [xs.len(), ys.len(), zs.len(), out.len()]
            .into_iter()
            .min()
            .unwrap();
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, Grad::default());
            }
            self.slice_size = size;
        }
        let mut v = SlotArray(&mut self.slots);
        for op in self.tape.iter_asm() {
            match op {
                AsmOp::Input(out, j) => {
                    for i in 0..size {
                        v[out][i] = match j {
                            0 => Grad::new(xs[i], 1.0, 0.0, 0.0),
                            1 => Grad::new(ys[i], 0.0, 1.0, 0.0),
                            2 => Grad::new(zs[i], 0.0, 0.0, 1.0),
                            _ => panic!("Invalid input: {}", i),
                        }
                    }
                }
                AsmOp::NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                AsmOp::AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                AsmOp::RecipReg(out, arg) => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                AsmOp::SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                AsmOp::SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                AsmOp::CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                AsmOp::AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                AsmOp::MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm.into();
                    }
                }
                AsmOp::DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                AsmOp::DivImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                AsmOp::SubImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                AsmOp::SubRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                AsmOp::MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                AsmOp::MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                AsmOp::AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                AsmOp::MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                AsmOp::DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                AsmOp::SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                AsmOp::MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                AsmOp::MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                AsmOp::CopyImm(out, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                AsmOp::Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                AsmOp::Store(out, mem) => {
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
    crate::grad_tests!(AsmGradEval);
    crate::interval_tests!(AsmIntervalEval);
    crate::float_slice_tests!(AsmFloatSliceEval);
    crate::point_tests!(AsmPointEval);
}
