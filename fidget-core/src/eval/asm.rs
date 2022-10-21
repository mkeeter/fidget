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
            use AsmOp::*;
            match op {
                Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RecipReg(out, arg) => {
                    v[out] = v[arg].recip();
                }
                SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                SquareReg(out, arg) => {
                    v[out] = v[arg].square();
                }
                CopyReg(out, arg) => v[out] = v[arg],
                AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm.into();
                }
                MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm.into();
                }
                DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm.into();
                }
                DivImmReg(out, arg, imm) => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                SubImmReg(out, arg, imm) => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm.into();
                }
                MinRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                MaxRegImm(out, arg, imm) => {
                    let (value, choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AddRegReg(out, lhs, rhs) => v[out] = v[lhs] + v[rhs],
                MulRegReg(out, lhs, rhs) => v[out] = v[lhs] * v[rhs],
                DivRegReg(out, lhs, rhs) => v[out] = v[lhs] / v[rhs],
                SubRegReg(out, lhs, rhs) => v[out] = v[lhs] - v[rhs],
                MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                CopyImm(out, imm) => {
                    v[out] = imm.into();
                }
                Load(out, mem) => {
                    v[out] = v[mem];
                }
                Store(out, mem) => {
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

    fn from_tape_give(
        tape: Tape,
        _s: Self::Storage,
    ) -> (Self, Option<Self::Storage>) {
        (Self::from(tape), None)
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
            use AsmOp::*;
            match op {
                Input(out, i) => v[out][0..size].copy_from_slice(match i {
                    0 => xs,
                    1 => ys,
                    2 => zs,
                    _ => panic!("Invalid input: {}", i),
                }),
                NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RecipReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                DivImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                SubImmReg(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                SubRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                MinRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                MaxRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                CopyImm(out, imm) => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                Store(out, mem) => {
                    for i in 0..size {
                        v[mem][i] = v[out][i];
                    }
                }
            }
        }
        out.copy_from_slice(&self.slots[0][0..size])
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
            use AsmOp::*;
            match op {
                Input(out, i) => {
                    v[out] = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                NegReg(out, arg) => {
                    v[out] = -v[arg];
                }
                AbsReg(out, arg) => {
                    v[out] = v[arg].abs();
                }
                RecipReg(out, arg) => {
                    v[out] = 1.0 / v[arg];
                }
                SqrtReg(out, arg) => {
                    v[out] = v[arg].sqrt();
                }
                SquareReg(out, arg) => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                CopyReg(out, arg) => {
                    v[out] = v[arg];
                }
                AddRegImm(out, arg, imm) => {
                    v[out] = v[arg] + imm;
                }
                MulRegImm(out, arg, imm) => {
                    v[out] = v[arg] * imm;
                }
                DivRegImm(out, arg, imm) => {
                    v[out] = v[arg] / imm;
                }
                DivImmReg(out, arg, imm) => {
                    v[out] = imm / v[arg];
                }
                SubImmReg(out, arg, imm) => {
                    v[out] = imm - v[arg];
                }
                SubRegImm(out, arg, imm) => {
                    v[out] = v[arg] - imm;
                }
                MinRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a < imm {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if imm < a {
                        choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        choices[choice_index] |= Choice::Both;
                        imm
                    };
                    choice_index += 1;
                }
                MaxRegImm(out, arg, imm) => {
                    let a = v[arg];
                    v[out] = if a > imm {
                        choices[choice_index] |= Choice::Left;
                        a
                    } else if imm > a {
                        choices[choice_index] |= Choice::Right;
                        imm
                    } else {
                        choices[choice_index] |= Choice::Both;
                        imm
                    };
                    choice_index += 1;
                }
                AddRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] + v[rhs];
                }
                MulRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] * v[rhs];
                }
                DivRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] / v[rhs];
                }
                SubRegReg(out, lhs, rhs) => {
                    v[out] = v[lhs] - v[rhs];
                }
                MinRegReg(out, lhs, rhs) => {
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
                        b
                    };
                    choice_index += 1;
                }
                MaxRegReg(out, lhs, rhs) => {
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
                        b
                    };
                    choice_index += 1;
                }
                CopyImm(out, imm) => {
                    v[out] = imm;
                }
                Load(out, mem) => {
                    v[out] = v[mem];
                }
                Store(out, mem) => {
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
            use AsmOp::*;
            match op {
                Input(out, j) => {
                    for i in 0..size {
                        v[out][i] = match j {
                            0 => Grad::new(xs[i], 1.0, 0.0, 0.0),
                            1 => Grad::new(ys[i], 0.0, 1.0, 0.0),
                            2 => Grad::new(zs[i], 0.0, 0.0, 1.0),
                            _ => panic!("Invalid input: {}", i),
                        }
                    }
                }
                NegReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                AbsReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                RecipReg(out, arg) => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                SqrtReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                SquareReg(out, arg) => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                CopyReg(out, arg) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                }
                AddRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                MulRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm.into();
                    }
                }
                DivRegImm(out, arg, imm) => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                DivImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                SubImmReg(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                SubRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                MinRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                MaxRegImm(out, arg, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                AddRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                MulRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                DivRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                SubRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                MinRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                MaxRegReg(out, lhs, rhs) => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                CopyImm(out, imm) => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Load(out, mem) => {
                    for i in 0..size {
                        v[out][i] = v[mem][i];
                    }
                }
                Store(out, mem) => {
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
mod tests {
    use super::*;
    use crate::{context::Context, eval::grad::GradEval};

    #[test]
    fn test_grad() {
        let mut ctx = Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let tape = ctx.get_tape(x, u8::MAX);

        let mut eval = GradEval::<AsmGradEval>::from(tape);
        assert_eq!(eval.eval_f(0.0, 0.0, 0.0), Grad::new(0.0, 1.0, 0.0, 0.0));

        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let sum = ctx.add(x2, y2).unwrap();
        let sqrt = ctx.sqrt(sum).unwrap();
        let half = ctx.constant(0.5);
        let sub = ctx.sub(sqrt, half).unwrap();
        let tape = ctx.get_tape(sub, u8::MAX);

        let mut eval = GradEval::<AsmGradEval>::from(tape);
        assert_eq!(eval.eval_f(1.0, 0.0, 0.0), Grad::new(0.5, 1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(0.0, 1.0, 0.0), Grad::new(0.5, 0.0, 1.0, 0.0));
        assert_eq!(eval.eval_f(2.0, 0.0, 0.0), Grad::new(1.5, 1.0, 0.0, 0.0));
        assert_eq!(eval.eval_f(0.0, 2.0, 0.0), Grad::new(1.5, 0.0, 1.0, 0.0));
    }
}
