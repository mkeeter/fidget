use crate::{
    asm::AsmOp,
    eval::{
        float_slice::{FloatSliceEvalT, FloatSliceFuncT},
        grad_slice::{Grad, GradSliceEvalT, GradSliceFuncT},
        interval::{Interval, IntervalEvalT, IntervalFuncT},
        point::{PointEvalT, PointFuncT},
        Choice, EvalFamily,
    },
    tape::Tape,
};

////////////////////////////////////////////////////////////////////////////////

/// Generic `struct` which wraps a `Tape` and converts it to an `Asm*Eval`
pub struct AsmFunc {
    tape: Tape,
}

/// Family of evaluators that use a local interpreter
pub enum AsmFamily {}

impl EvalFamily for AsmFamily {
    /// This is interpreted, so we can use the maximum number of registers
    const REG_LIMIT: u8 = u8::MAX;

    type IntervalFunc = AsmFunc;
    type FloatSliceFunc = AsmFunc;
    type PointFunc = AsmFunc;
    type GradSliceFunc = AsmFunc;
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

impl IntervalFuncT for AsmFunc {
    type Evaluator = AsmIntervalEval;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmIntervalEval::new(self.tape.clone())
    }

    fn from_tape(tape: Tape) -> Self {
        AsmFunc { tape }
    }
}

/// Interval evaluator for a slice of [`AsmOp`]
pub struct AsmIntervalEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Interval>,
}

impl AsmIntervalEval {
    pub fn new(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            slots: vec![Interval::from(std::f32::NAN); tape.slot_count()],
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
                    let a = v[arg];
                    v[out] = if a.upper() < 0.0 {
                        Interval::new(a.upper().powi(2), a.lower().powi(2))
                    } else if v[arg].lower() > 0.0 {
                        Interval::new(a.lower().powi(2), a.upper().powi(2))
                    } else {
                        Interval::new(0.0, a.lower().max(a.upper()).powi(2))
                    };
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

impl FloatSliceFuncT for AsmFunc {
    type Evaluator = AsmFloatSliceEval;
    type Storage = ();

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmFloatSliceEval::new(self.tape.clone())
    }

    fn from_tape(tape: Tape) -> Self {
        AsmFunc { tape }
    }

    fn from_tape_give(
        tape: Tape,
        _s: Self::Storage,
    ) -> (Self, Option<Self::Storage>) {
        (AsmFunc { tape }, None)
    }

    fn take(self) -> Option<()> {
        // There is no storage, so you can always take it
        Some(())
    }
}

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmFloatSliceEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Vec<f32>>,
    /// Current slice size in `self.slots`
    slice_size: usize,
}

impl AsmFloatSliceEval {
    pub fn new(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            slots: vec![vec![]; tape.slot_count()],
            slice_size: 0,
        }
    }
}

impl FloatSliceEvalT for AsmFloatSliceEval {
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

////////////////////////////////////////////////////////////////////////////////

impl PointFuncT for AsmFunc {
    type Evaluator = AsmPointEval;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmPointEval::new(self.tape.clone())
    }
    fn from_tape(tape: Tape) -> Self {
        AsmFunc { tape }
    }
}

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmPointEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<f32>,
}

impl AsmPointEval {
    pub fn new(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            slots: vec![std::f32::NAN; tape.slot_count()],
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

impl GradSliceFuncT for AsmFunc {
    type Evaluator = AsmGradSliceEval;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmGradSliceEval::new(self.tape.clone())
    }
    fn from_tape(tape: Tape) -> Self {
        AsmFunc { tape }
    }
}

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmGradSliceEval {
    /// Instruction tape, in reverse-evaluation order
    tape: Tape,
    /// Workspace for data
    slots: Vec<Vec<Grad>>,
    slice_size: usize,
}

impl AsmGradSliceEval {
    pub fn new(tape: Tape) -> Self {
        Self {
            tape: tape.clone(),
            slots: vec![vec![]; tape.slot_count()],
            slice_size: 0,
        }
    }
}

impl GradSliceEvalT for AsmGradSliceEval {
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
