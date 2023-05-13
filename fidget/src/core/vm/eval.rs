use crate::{
    eval::{
        bulk::{BulkEvaluator, BulkEvaluatorData},
        tracing::{TracingEvaluator, TracingEvaluatorData},
        types::{Grad, Interval},
        Choice, EvaluatorStorage, Family,
    },
    vm::{Op, Tape, TapeData},
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
pub struct AsmEval(Tape<Eval>);

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
    fn prepare(&mut self, tape: &TapeData<Eval>) {
        assert!(tape.reg_limit() == u8::MAX);

        let slot_count = tape.slot_count();
        self.slots.resize(slot_count, T::from(std::f32::NAN));
        self.slots.fill(T::from(std::f32::NAN));
    }
}

impl EvaluatorStorage<Eval> for AsmEval {
    type Storage = ();
    fn new_with_storage(tape: &Tape<Eval>, _storage: ()) -> Self {
        Self(tape.clone())
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
        assert_eq!(vars.len(), self.0.tape.var_count());

        let mut choice_index = 0;
        let mut v = SlotArray(&mut data.slots);
        for op in self.0.iter_asm() {
            match op {
                Op::Input { out, input } => {
                    v[out] = match input {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {input}"),
                    }
                }
                Op::Var { out, var } => {
                    v[out] = vars[var as usize].into();
                }
                Op::NegReg { out, arg } => {
                    v[out] = -v[arg];
                }
                Op::AbsReg { out, arg } => {
                    v[out] = v[arg].abs();
                }
                Op::RecipReg { out, arg } => {
                    v[out] = v[arg].recip();
                }
                Op::SqrtReg { out, arg } => {
                    v[out] = v[arg].sqrt();
                }
                Op::SquareReg { out, arg } => {
                    v[out] = v[arg].square();
                }
                Op::AddRegImm { out, arg, imm } => {
                    v[out] = v[arg] + imm.into();
                }
                Op::MulRegImm { out, arg, imm } => {
                    v[out] = v[arg] * imm.into();
                }
                Op::DivRegImm { out, arg, imm } => {
                    v[out] = v[arg] / imm.into();
                }
                Op::DivImmReg { out, arg, imm } => {
                    let imm: Interval = imm.into();
                    v[out] = imm / v[arg];
                }
                Op::SubImmReg { out, arg, imm } => {
                    v[out] = Interval::from(imm) - v[arg];
                }
                Op::SubRegImm { out, arg, imm } => {
                    v[out] = v[arg] - imm.into();
                }
                Op::MinRegImm { out, arg, imm } => {
                    let (value, _choice) = v[arg].min_choice(imm.into());
                    v[out] = value;
                }
                Op::MaxRegImm { out, arg, imm } => {
                    let (value, _choice) = v[arg].max_choice(imm.into());
                    v[out] = value;
                }

                Op::MinRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[arg], Choice::Left),
                        Choice::Right => (imm.into(), Choice::Right),
                        Choice::Both => v[arg].min_choice(imm.into()),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MaxRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[arg], Choice::Left),
                        Choice::Right => (imm.into(), Choice::Right),
                        Choice::Both => v[arg].max_choice(imm.into()),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MinRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[lhs], Choice::Left),
                        Choice::Right => (v[rhs], Choice::Right),
                        Choice::Both => v[lhs].min_choice(v[rhs]),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MaxRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[lhs], Choice::Left),
                        Choice::Right => (v[rhs], Choice::Right),
                        Choice::Both => v[lhs].max_choice(v[rhs]),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::AddRegReg { out, lhs, rhs } => v[out] = v[lhs] + v[rhs],
                Op::MulRegReg { out, lhs, rhs } => v[out] = v[lhs] * v[rhs],
                Op::DivRegReg { out, lhs, rhs } => v[out] = v[lhs] / v[rhs],
                Op::SubRegReg { out, lhs, rhs } => v[out] = v[lhs] - v[rhs],
                Op::MinRegReg { out, lhs, rhs } => {
                    let (value, _choice) = v[lhs].min_choice(v[rhs]);
                    v[out] = value;
                }
                Op::MaxRegReg { out, lhs, rhs } => {
                    let (value, _choice) = v[lhs].max_choice(v[rhs]);
                    v[out] = value;
                }
                Op::CopyImm { out, imm } => {
                    v[out] = imm.into();
                }
                Op::Load { reg, mem } => {
                    v[reg] = v[mem];
                }
                Op::Store { reg, mem } => {
                    v[mem] = v[reg];
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
        assert_eq!(vars.len(), self.0.tape.var_count());
        let mut choice_index = 0;
        let mut simplify = false;
        let mut v = SlotArray(&mut data.slots);
        for op in self.0.iter_asm() {
            match op {
                Op::Input { out, input } => {
                    v[out] = match input {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => panic!("Invalid input: {input}"),
                    }
                }
                Op::Var { out, var } => v[out] = vars[var as usize],
                Op::NegReg { out, arg } => {
                    v[out] = -v[arg];
                }
                Op::AbsReg { out, arg } => {
                    v[out] = v[arg].abs();
                }
                Op::RecipReg { out, arg } => {
                    v[out] = 1.0 / v[arg];
                }
                Op::SqrtReg { out, arg } => {
                    v[out] = v[arg].sqrt();
                }
                Op::SquareReg { out, arg } => {
                    let s = v[arg];
                    v[out] = s * s;
                }
                Op::AddRegImm { out, arg, imm } => {
                    v[out] = v[arg] + imm;
                }
                Op::MulRegImm { out, arg, imm } => {
                    v[out] = v[arg] * imm;
                }
                Op::DivRegImm { out, arg, imm } => {
                    v[out] = v[arg] / imm;
                }
                Op::DivImmReg { out, arg, imm } => {
                    v[out] = imm / v[arg];
                }
                Op::SubImmReg { out, arg, imm } => {
                    v[out] = imm - v[arg];
                }
                Op::SubRegImm { out, arg, imm } => {
                    v[out] = v[arg] - imm;
                }
                Op::MinRegImm { out, arg, imm } => {
                    let (value, _choice) = min_choice(v[arg], imm);
                    v[out] = value;
                }
                Op::MaxRegImm { out, arg, imm } => {
                    let (value, _choice) = max_choice(v[arg], imm);
                    v[out] = value;
                }

                Op::MinRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[arg], Choice::Left),
                        Choice::Right => (imm.into(), Choice::Right),
                        Choice::Both => min_choice(v[arg], imm.into()),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MaxRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[arg], Choice::Left),
                        Choice::Right => (imm.into(), Choice::Right),
                        Choice::Both => max_choice(v[arg], imm),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MinRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[lhs], Choice::Left),
                        Choice::Right => (v[rhs], Choice::Right),
                        Choice::Both => min_choice(v[lhs], v[rhs]),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }

                Op::MaxRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    let (value, choice) = match self.0.choices[choice as usize]
                    {
                        Choice::Left => (v[lhs], Choice::Left),
                        Choice::Right => (v[rhs], Choice::Right),
                        Choice::Both => max_choice(v[lhs], v[rhs]),
                        Choice::Unknown => {
                            panic!("invalid choice in evaluation")
                        }
                    };
                    v[out] = value;
                    choices[choice_index] |= choice;
                    choice_index += 1;
                    simplify |= choice != Choice::Both;
                }
                Op::AddRegReg { out, lhs, rhs } => {
                    v[out] = v[lhs] + v[rhs];
                }
                Op::MulRegReg { out, lhs, rhs } => {
                    v[out] = v[lhs] * v[rhs];
                }
                Op::DivRegReg { out, lhs, rhs } => {
                    v[out] = v[lhs] / v[rhs];
                }
                Op::SubRegReg { out, lhs, rhs } => {
                    v[out] = v[lhs] - v[rhs];
                }
                Op::MinRegReg { out, lhs, rhs } => {
                    let (value, _choice) = min_choice(v[lhs], v[rhs]);
                    v[out] = value;
                }
                Op::MaxRegReg { out, lhs, rhs } => {
                    let (value, _choice) = max_choice(v[lhs], v[rhs]);
                    v[out] = value;
                }
                Op::CopyImm { out, imm } => {
                    v[out] = imm;
                }
                Op::Load { reg, mem } => {
                    v[reg] = v[mem];
                }
                Op::Store { reg, mem } => {
                    v[mem] = v[reg];
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
    fn prepare(&mut self, tape: &TapeData<Eval>, size: usize) {
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
        assert_eq!(vars.len(), self.0.tape.var_count());
        assert_eq!(data.slots.len(), self.0.tape.slot_count());

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.0.iter_asm() {
            match op {
                Op::Input { out, input } => {
                    v[out][0..size].copy_from_slice(match input {
                        0 => xs,
                        1 => ys,
                        2 => zs,
                        _ => panic!("Invalid input: {input}"),
                    })
                }
                Op::Var { out, var } => {
                    v[out][0..size].fill(vars[var as usize])
                }
                Op::NegReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                Op::AbsReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                Op::RecipReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = 1.0 / v[arg][i];
                    }
                }
                Op::SqrtReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                Op::SquareReg { out, arg } => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                Op::AddRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm;
                    }
                }
                Op::MulRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm;
                    }
                }
                Op::DivRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm;
                    }
                }
                Op::DivImmReg { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                Op::SubImmReg { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                Op::SubRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                Op::MinRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                Op::MaxRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                Op::AddRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                Op::MulRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                Op::DivRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                Op::SubRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                Op::MinRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                Op::MaxRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                Op::CopyImm { out, imm } => {
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Op::Load { reg, mem } => {
                    for i in 0..size {
                        v[reg][i] = v[mem][i];
                    }
                }
                Op::Store { reg, mem } => {
                    for i in 0..size {
                        v[mem][i] = v[reg][i];
                    }
                }
                Op::MinRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[arg][i],
                            Choice::Right => imm,
                            Choice::Both => v[arg][i].min(imm),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MaxRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[arg][i],
                            Choice::Right => imm,
                            Choice::Both => v[arg][i].max(imm),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MinRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[lhs][i],
                            Choice::Right => v[rhs][i],
                            Choice::Both => v[rhs][i].min(v[rhs][i]),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MaxRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[lhs][i],
                            Choice::Right => v[rhs][i],
                            Choice::Both => v[rhs][i].max(v[rhs][i]),
                            Choice::Unknown => panic!(),
                        };
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
        assert_eq!(vars.len(), self.0.tape.var_count());
        assert_eq!(data.slots.len(), self.0.tape.slot_count());

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.0.iter_asm() {
            match op {
                Op::Input { out, input } => {
                    for i in 0..size {
                        v[out][i] = match input {
                            0 => Grad::new(xs[i], 1.0, 0.0, 0.0),
                            1 => Grad::new(ys[i], 0.0, 1.0, 0.0),
                            2 => Grad::new(zs[i], 0.0, 0.0, 1.0),
                            _ => panic!("Invalid input: {}", i),
                        }
                    }
                }
                Op::Var { out, var } => {
                    // TODO: error handling?
                    v[out][0..size].fill(Grad::new(
                        vars[var as usize],
                        0.0,
                        0.0,
                        0.0,
                    ));
                }
                Op::NegReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = -v[arg][i];
                    }
                }
                Op::AbsReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].abs();
                    }
                }
                Op::RecipReg { out, arg } => {
                    let one: Grad = 1.0.into();
                    for i in 0..size {
                        v[out][i] = one / v[arg][i];
                    }
                }
                Op::SqrtReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i].sqrt();
                    }
                }
                Op::SquareReg { out, arg } => {
                    for i in 0..size {
                        let s = v[arg][i];
                        v[out][i] = s * s;
                    }
                }
                Op::AddRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] + imm.into();
                    }
                }
                Op::MulRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] * imm.into();
                    }
                }
                Op::DivRegImm { out, arg, imm } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i] / imm.into();
                    }
                }
                Op::DivImmReg { out, arg, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm / v[arg][i];
                    }
                }
                Op::SubImmReg { out, arg, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm - v[arg][i];
                    }
                }
                Op::SubRegImm { out, arg, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i] - imm;
                    }
                }
                Op::MinRegImm { out, arg, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].min(imm);
                    }
                }
                Op::MaxRegImm { out, arg, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = v[arg][i].max(imm);
                    }
                }
                Op::MinRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[arg][i],
                            Choice::Right => imm,
                            Choice::Both => v[arg][i].min(imm),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MaxRegImmChoice {
                    out,
                    arg,
                    imm,
                    choice,
                } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[arg][i],
                            Choice::Right => imm,
                            Choice::Both => v[arg][i].max(imm),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MinRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[lhs][i],
                            Choice::Right => v[rhs][i],
                            Choice::Both => v[lhs][i].min(v[rhs][i]),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::MaxRegRegChoice {
                    out,
                    lhs,
                    rhs,
                    choice,
                } => {
                    for i in 0..size {
                        v[out][i] = match self.0.choices[choice as usize] {
                            Choice::Left => v[lhs][i],
                            Choice::Right => v[rhs][i],
                            Choice::Both => v[lhs][i].max(v[rhs][i]),
                            Choice::Unknown => panic!(),
                        };
                    }
                }
                Op::AddRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] + v[rhs][i];
                    }
                }
                Op::MulRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] * v[rhs][i];
                    }
                }
                Op::DivRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] / v[rhs][i];
                    }
                }
                Op::SubRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i] - v[rhs][i];
                    }
                }
                Op::MinRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].min(v[rhs][i]);
                    }
                }
                Op::MaxRegReg { out, lhs, rhs } => {
                    for i in 0..size {
                        v[out][i] = v[lhs][i].max(v[rhs][i]);
                    }
                }
                Op::CopyImm { out, imm } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[out][i] = imm;
                    }
                }
                Op::Load { reg, mem } => {
                    for i in 0..size {
                        v[reg][i] = v[mem][i];
                    }
                }
                Op::Store { reg, mem } => {
                    for i in 0..size {
                        v[mem][i] = v[reg][i];
                    }
                }
            }
        }
        out.copy_from_slice(&data.slots[0][0..size])
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Calculate the `min` value and a `Choice`, given two floating-point values
///
/// Unlike `f32::min`, this value will be `NAN` if _either_ input is `NAN`
fn min_choice(a: f32, b: f32) -> (f32, Choice) {
    if a < b {
        (a, Choice::Left)
    } else if b < a {
        (b, Choice::Right)
    } else if a.is_nan() || b.is_nan() {
        (std::f32::NAN, Choice::Both)
    } else {
        (b, Choice::Both)
    }
}

/// Calculate the `max` value and a `Choice`, given two floating-point values
///
/// Unlike `f32::max`, this value will be `NAN` if _either_ input is `NAN`
fn max_choice(a: f32, b: f32) -> (f32, Choice) {
    if a > b {
        (a, Choice::Left)
    } else if b > a {
        (b, Choice::Right)
    } else if a.is_nan() || b.is_nan() {
        (std::f32::NAN, Choice::Both)
    } else {
        (b, Choice::Both)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    crate::grad_slice_tests!(Eval);
    crate::interval_tests!(Eval);
    crate::float_slice_tests!(Eval);
    crate::point_tests!(Eval);
}
