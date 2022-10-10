use crate::{
    asm::AsmOp,
    eval::{
        float_slice::{FloatSliceEvalT, FloatSliceFuncT},
        interval::{Interval, IntervalEvalT, IntervalFuncT},
        point::{PointEvalT, PointFuncT},
        Choice, EvalFamily,
    },
    tape::Tape,
};

////////////////////////////////////////////////////////////////////////////////

/// Generic `struct` which wraps a `Tape` and converts it to an `Asm*Eval`
pub struct AsmFunc<'a> {
    tape: &'a Tape,
}

/// Family of evaluators that use a local interpreter
pub struct AsmFamily<'a> {
    _p: std::marker::PhantomData<&'a ()>,
}

impl<'a> EvalFamily for AsmFamily<'a> {
    /// This is interpreted, so we can use the maximum number of registers
    const REG_LIMIT: u8 = u8::MAX;

    type Recurse<'b> = AsmFamily<'b>;

    type IntervalFunc = AsmFunc<'a>;
    type FloatSliceFunc = AsmFunc<'a>;
    type PointFunc = AsmFunc<'a>;
}

////////////////////////////////////////////////////////////////////////////////

impl<'a> IntervalFuncT for AsmFunc<'a> {
    type Evaluator = AsmIntervalEval<'a>;
    type Recurse<'b> = AsmFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmIntervalEval::new(self.tape)
    }

    fn from_tape(tape: &Tape) -> Self::Recurse<'_> {
        AsmFunc { tape }
    }
}

/// Interval evaluator for a slice of [`AsmOp`]
pub struct AsmIntervalEval<'t> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'t Tape,
    /// Workspace for data
    slots: Vec<Interval>,
}

impl<'a> AsmIntervalEval<'a> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            slots: vec![], // dynamically resized at runtime
        }
    }

    fn v<I: Into<usize>>(&mut self, i: I) -> &mut Interval {
        let i: usize = i.into();
        if i >= self.slots.len() {
            self.slots.resize(i + 1, Interval::from(std::f32::NAN));
        }
        &mut self.slots[i]
    }
}

impl<'a> IntervalEvalT for AsmIntervalEval<'a> {
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
        for op in self.tape.iter_asm() {
            use AsmOp::*;
            match op {
                Input(out, i) => {
                    *self.v(out) = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                NegReg(out, arg) => {
                    *self.v(out) = -*self.v(arg);
                }
                AbsReg(out, arg) => {
                    *self.v(out) = self.v(arg).abs();
                }
                RecipReg(out, arg) => {
                    *self.v(out) = self.v(arg).recip();
                }
                SqrtReg(out, arg) => {
                    *self.v(out) = self.v(arg).sqrt();
                }
                SquareReg(out, arg) => {
                    *self.v(out) = *self.v(arg) * *self.v(arg)
                }
                CopyReg(out, arg) => *self.v(out) = *self.v(arg),
                AddRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) + imm.into();
                }
                MulRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) * imm.into();
                }
                SubImmReg(out, arg, imm) => {
                    *self.v(out) = Interval::from(imm) - *self.v(arg);
                }
                SubRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) - imm.into();
                }
                MinRegImm(out, arg, imm) => {
                    let (value, choice) = self.v(arg).min_choice(imm.into());
                    *self.v(out) = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                MaxRegImm(out, arg, imm) => {
                    let (value, choice) = self.v(arg).max_choice(imm.into());
                    *self.v(out) = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                AddRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) + *self.v(rhs)
                }
                MulRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) * *self.v(rhs)
                }
                SubRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) - *self.v(rhs)
                }
                MinRegReg(out, lhs, rhs) => {
                    let (value, choice) = self.v(lhs).min_choice(*self.v(rhs));
                    *self.v(out) = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = self.v(lhs).max_choice(*self.v(rhs));
                    *self.v(out) = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                }
                CopyImm(out, imm) => {
                    *self.v(out) = imm.into();
                }
                Load(out, mem) => {
                    *self.v(out) = self.slots[mem as usize];
                }
                Store(out, mem) => {
                    *self.v(mem as usize) = *self.v(out);
                }
            }
        }
        self.slots[0]
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<'a> FloatSliceFuncT for AsmFunc<'a> {
    type Evaluator = AsmFloatSliceEval<'a>;
    type Recurse<'b> = AsmFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmFloatSliceEval::new(self.tape)
    }

    fn from_tape(tape: &Tape) -> Self::Recurse<'_> {
        AsmFunc { tape }
    }
}

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmFloatSliceEval<'t> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'t Tape,
    /// Workspace for data
    slots: Vec<Vec<f32>>,
    /// Current slice size in `self.slots`
    slice_size: usize,
}

impl<'a> AsmFloatSliceEval<'a> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            slots: vec![], // dynamically resized at runtime
            slice_size: 0,
        }
    }
    fn v<I: Into<usize>>(&mut self, i: I) -> &mut [f32] {
        let i: usize = i.into();
        if i >= self.slots.len() {
            self.slots.resize_with(i + 1, Vec::new);
        }
        if self.slots[i].len() < self.slice_size {
            self.slots[i].resize(self.slice_size, std::f32::NAN);
        }
        &mut self.slots[i]
    }
}

impl<'a> FloatSliceEvalT for AsmFloatSliceEval<'a> {
    fn eval_s(&mut self, xs: &[f32], ys: &[f32], zs: &[f32], out: &mut [f32]) {
        self.slice_size = xs.len().min(ys.len()).min(zs.len()).min(out.len());
        for op in self.tape.iter_asm() {
            use AsmOp::*;
            match op {
                Input(out, i) => self.v(out).copy_from_slice(match i {
                    0 => xs,
                    1 => ys,
                    2 => zs,
                    _ => panic!("Invalid input: {}", i),
                }),
                NegReg(out, arg) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = -self.v(arg)[i];
                    }
                }
                AbsReg(out, arg) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i].abs();
                    }
                }
                RecipReg(out, arg) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = 1.0 / self.v(arg)[i];
                    }
                }
                SqrtReg(out, arg) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i].sqrt();
                    }
                }
                SquareReg(out, arg) => {
                    for i in 0..self.slice_size {
                        let s = self.v(arg)[i];
                        self.v(out)[i] = s * s;
                    }
                }
                CopyReg(out, arg) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i];
                    }
                }
                AddRegImm(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i] + imm;
                    }
                }
                MulRegImm(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i] * imm;
                    }
                }
                SubImmReg(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = imm - self.v(arg)[i];
                    }
                }
                SubRegImm(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i] - imm;
                    }
                }
                MinRegImm(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i].min(imm);
                    }
                }
                MaxRegImm(out, arg, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(arg)[i].max(imm);
                    }
                }
                AddRegReg(out, lhs, rhs) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(lhs)[i] + self.v(rhs)[i];
                    }
                }
                MulRegReg(out, lhs, rhs) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(lhs)[i] * self.v(rhs)[i];
                    }
                }
                SubRegReg(out, lhs, rhs) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(lhs)[i] - self.v(rhs)[i];
                    }
                }
                MinRegReg(out, lhs, rhs) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(lhs)[i].min(self.v(rhs)[i]);
                    }
                }
                MaxRegReg(out, lhs, rhs) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(lhs)[i].max(self.v(rhs)[i]);
                    }
                }
                CopyImm(out, imm) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = imm;
                    }
                }
                Load(out, mem) => {
                    for i in 0..self.slice_size {
                        self.v(out)[i] = self.v(mem as usize)[i];
                    }
                }
                Store(out, mem) => {
                    for i in 0..self.slice_size {
                        self.v(mem as usize)[i] = self.v(out)[i];
                    }
                }
            }
        }
        out.copy_from_slice(&self.slots[0])
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<'a> PointFuncT for AsmFunc<'a> {
    type Evaluator = AsmPointEval<'a>;
    type Recurse<'b> = AsmFunc<'b>;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmPointEval::new(self.tape)
    }
    fn from_tape(tape: &Tape) -> Self::Recurse<'_> {
        AsmFunc { tape }
    }
}

/// Float-point interpreter-style evaluator for a tape of [`AsmOp`]
pub struct AsmPointEval<'t> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'t Tape,
    /// Workspace for data
    slots: Vec<f32>,
}

impl<'a> AsmPointEval<'a> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            slots: vec![], // dynamically resized at runtime
        }
    }
    fn v<I: Into<usize>>(&mut self, i: I) -> &mut f32 {
        let i: usize = i.into();
        if i >= self.slots.len() {
            self.slots.resize(i + 1, std::f32::NAN);
        }
        &mut self.slots[i]
    }
}

impl<'a> PointEvalT for AsmPointEval<'a> {
    fn eval_p(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        choices: &mut [Choice],
    ) -> f32 {
        let mut choice_index = 0;
        for op in self.tape.iter_asm() {
            use AsmOp::*;
            match op {
                Input(out, i) => {
                    *self.v(out) = match i {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {}", i),
                    }
                }
                NegReg(out, arg) => {
                    *self.v(out) = -*self.v(arg);
                }
                AbsReg(out, arg) => {
                    *self.v(out) = self.v(arg).abs();
                }
                RecipReg(out, arg) => {
                    *self.v(out) = 1.0 / *self.v(arg);
                }
                SqrtReg(out, arg) => {
                    *self.v(out) = self.v(arg).sqrt();
                }
                SquareReg(out, arg) => {
                    let s = *self.v(arg);
                    *self.v(out) = s * s;
                }
                CopyReg(out, arg) => {
                    *self.v(out) = *self.v(arg);
                }
                AddRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) + imm;
                }
                MulRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) * imm;
                }
                SubImmReg(out, arg, imm) => {
                    *self.v(out) = imm - *self.v(arg);
                }
                SubRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) - imm;
                }
                MinRegImm(out, arg, imm) => {
                    let a = *self.v(arg);
                    *self.v(out) = if a < imm {
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
                    let a = *self.v(arg);
                    *self.v(out) = if a > imm {
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
                    *self.v(out) = *self.v(lhs) + *self.v(rhs);
                }
                MulRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) * *self.v(rhs);
                }
                SubRegReg(out, lhs, rhs) => {
                    *self.v(out) = *self.v(lhs) - *self.v(rhs);
                }
                MinRegReg(out, lhs, rhs) => {
                    let a = *self.v(lhs);
                    let b = *self.v(rhs);
                    *self.v(out) = if a < b {
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
                    let a = *self.v(lhs);
                    let b = *self.v(rhs);
                    *self.v(out) = if a > b {
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
                    *self.v(out) = imm;
                }
                Load(out, mem) => {
                    *self.v(out) = *self.v(mem as usize);
                }
                Store(out, mem) => {
                    *self.v(mem as usize) = *self.v(out);
                }
            }
        }
        self.slots[0]
    }
}
