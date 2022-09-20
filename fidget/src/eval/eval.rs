use crate::{
    asm::AsmOp,
    eval::{
        Choice, EvalFamily, FloatSliceEval, FloatSliceFunc, Interval,
        IntervalEval, IntervalFunc,
    },
    tape::Tape,
};

////////////////////////////////////////////////////////////////////////////////

/// Generic `struct` which wraps a `Tape` and converts it to an `Asm*Eval`
pub struct AsmFunc<'a> {
    tape: &'a Tape,
}

/// Family of evaluators that use a local interpreter
pub enum AsmFamily {}

impl<'a> EvalFamily<'a> for AsmFamily {
    type IntervalFunc = AsmFunc<'a>;
    type FloatSliceFunc = AsmFunc<'a>;
    fn from_tape_i(t: &Tape) -> AsmFunc {
        AsmFunc { tape: t }
    }
    fn from_tape_s(t: &Tape) -> AsmFunc {
        AsmFunc { tape: t }
    }
}

////////////////////////////////////////////////////////////////////////////////

impl<'a> IntervalFunc<'a> for AsmFunc<'a> {
    type Evaluator = AsmIntervalEval<'a>;
    fn get_evaluator(&self) -> Self::Evaluator {
        AsmIntervalEval::new(self.tape)
    }
}

impl<'a> AsmFunc<'a> {
    pub fn from_tape(tape: &Tape) -> AsmFunc {
        AsmFunc { tape }
    }
}

/// Interval evaluator for a slice of [`AsmOp`]
pub struct AsmIntervalEval<'t> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'t Tape,
    /// Workspace for data
    slots: Vec<Interval>,
    /// Choice array, in evaluation (forward) order
    choices: Vec<Choice>,
    /// Raw choice array, as bitfields
    choices_raw: Vec<u8>,
}

impl<'a> AsmIntervalEval<'a> {
    pub fn new(tape: &'a Tape) -> Self {
        Self {
            tape,
            slots: vec![], // dynamically resized at runtime
            choices: vec![Choice::Unknown; tape.choice_count()],
            choices_raw: vec![0; tape.choice_count()],
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

impl<'a> IntervalEval<'a> for AsmIntervalEval<'a> {
    fn simplify(&self) -> Tape {
        self.tape.simplify(&self.choices)
    }
    fn reset_choices(&mut self) {
        self.choices.fill(Choice::Unknown);
    }
    fn load_choices(&mut self) {
        // Nothing to do here
    }
    fn eval_i_inner<I: Into<Interval>>(
        &mut self,
        x: I,
        y: I,
        z: I,
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
                    self.choices_raw[choice_index] |= choice as u8;
                    choice_index += 1;
                }
                MaxRegImm(out, arg, imm) => {
                    let (value, choice) = self.v(arg).max_choice(imm.into());
                    *self.v(out) = value;
                    self.choices_raw[choice_index] |= choice as u8;
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
                    self.choices_raw[choice_index] |= choice as u8;
                    choice_index += 1;
                }
                MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = self.v(lhs).max_choice(*self.v(rhs));
                    *self.v(out) = value;
                    self.choices_raw[choice_index] |= choice as u8;
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

impl<'a> FloatSliceFunc<'a> for AsmFunc<'a> {
    type Evaluator = AsmFloatSliceEval<'a>;

    fn get_evaluator(&self) -> Self::Evaluator {
        AsmFloatSliceEval::new(self.tape)
    }
}

/// Interval evaluator for a slice of [`AsmOp`]
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

impl<'a> FloatSliceEval<'a> for AsmFloatSliceEval<'a> {
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
    }
}
