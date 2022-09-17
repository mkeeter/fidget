use crate::{
    asm::AsmOp,
    eval::{Choice, EvalMath, FloatEval, Interval, IntervalEval},
};

/// Evaluator for a slice of [`AsmOp`]
pub struct AsmEval<'a, T> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'a [AsmOp],
    choices: Vec<Choice>,
    /// Workspace for data
    slots: Vec<T>,
}

impl<'a, T: EvalMath> AsmEval<'a, T> {
    pub fn new(tape: &'a [AsmOp]) -> Self {
        Self {
            tape,
            slots: vec![],
            choices: vec![],
        }
    }
    fn v(&mut self, i: u8) -> &mut T {
        if i as usize >= self.slots.len() {
            self.slots.resize(i as usize + 1, T::from(std::f32::NAN));
        }
        &mut self.slots[i as usize]
    }
    pub fn eval(&mut self, x: T, y: T, z: T) -> T {
        for &op in self.tape.iter().rev() {
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
                    *self.v(out) = T::from(imm) - *self.v(arg);
                }
                SubRegImm(out, arg, imm) => {
                    *self.v(out) = *self.v(arg) - imm.into();
                }
                MinRegImm(out, arg, imm) => {
                    let (value, choice) = self.v(arg).min_choice(imm.into());
                    *self.v(out) = value;
                    self.choices.push(choice);
                }
                MaxRegImm(out, arg, imm) => {
                    let (value, choice) = self.v(arg).max_choice(imm.into());
                    *self.v(out) = value;
                    self.choices.push(choice);
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
                    self.choices.push(choice);
                }
                MaxRegReg(out, lhs, rhs) => {
                    let (value, choice) = self.v(lhs).max_choice(*self.v(rhs));
                    *self.v(out) = value;
                    self.choices.push(choice);
                }
                CopyImm(out, imm) => {
                    *self.v(out) = imm.into();
                }
                Load(out, mem) => {
                    *self.v(out) = self.slots[mem as usize];
                }
                Store(out, mem) => {
                    if mem as usize >= self.slots.len() {
                        self.slots
                            .resize(mem as usize + 1, T::from(std::f32::NAN));
                    }
                    self.slots[mem as usize] = *self.v(out);
                }
            }
        }
        self.slots[0]
    }
}

impl<'a> IntervalEval for AsmEval<'a, Interval> {
    fn choices(&self) -> &[Choice] {
        &self.choices
    }
    fn eval(&mut self, x: Interval, y: Interval, z: Interval) -> Interval {
        AsmEval::eval(self, x, y, z)
    }
}

impl<'a> FloatEval for AsmEval<'a, f32> {
    fn eval(&mut self, x: f32, y: f32, z: f32) -> f32 {
        AsmEval::eval(self, x, y, z)
    }
}
