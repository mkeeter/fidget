use crate::{
    asm::AsmOp,
    eval::{
        Choice, EvalMath, FloatEval, FloatFunc, Interval, IntervalEval,
        IntervalFunc,
    },
    tape::Tape,
};

pub struct InterpreterHandle<'a> {
    tape: &'a Tape,
}

impl<'a> IntervalFunc<'a> for InterpreterHandle<'a> {
    type Evaluator<'b> = AsmEval<'b, Interval> where Self: 'b;
    type Recurse<'b> = InterpreterHandle<'b>;
    fn get_evaluator(&self) -> Self::Evaluator<'_> {
        self.tape.get_evaluator()
    }
    fn from_tape(tape: &Tape) -> InterpreterHandle {
        InterpreterHandle { tape }
    }
}

impl<'a> FloatFunc<'a> for InterpreterHandle<'a> {
    type Evaluator<'b> = AsmEval<'b, f32> where Self: 'b;
    type Recurse<'b> = InterpreterHandle<'b>;

    fn get_evaluator(&self) -> Self::Evaluator<'_> {
        self.tape.get_evaluator()
    }
    fn from_tape(tape: &Tape) -> InterpreterHandle {
        InterpreterHandle { tape }
    }
}

impl<'a> IntervalEval<'a> for AsmEval<'a, Interval> {
    fn simplify(&self) -> Tape {
        self.tape.simplify(&self.choices)
    }
    fn eval_i(&mut self, x: Interval, y: Interval, z: Interval) -> Interval {
        AsmEval::eval(self, x, y, z)
    }
}

impl<'a> FloatEval<'a> for AsmEval<'a, f32> {
    fn eval_f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        AsmEval::eval(self, x, y, z)
    }
}

/// Evaluator for a slice of [`AsmOp`]
pub struct AsmEval<'t, T> {
    /// Instruction tape, in reverse-evaluation order
    tape: &'t Tape,
    /// Workspace for data
    slots: Vec<T>,
    /// Choice array, in evaluation (forward) order
    choices: Vec<Choice>,
}

impl<'a, T: EvalMath> AsmEval<'a, T> {
    pub fn new(tape: &'a Tape) -> Self {
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
        self.choices.clear();
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
