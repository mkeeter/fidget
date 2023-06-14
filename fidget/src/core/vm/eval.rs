use crate::{
    eval::{
        bulk::{BulkEvaluator, BulkEvaluatorData},
        tracing::{TracingEvaluator, TracingEvaluatorData},
        types::{Choice, Grad, Interval},
        EvaluatorStorage, Family,
    },
    vm::{Choices, Op, Tape, TapeData},
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
        choices: &mut Choices,
        data: &mut Self::Data,
    ) -> (Interval, bool) {
        let mut simplify = false;
        assert_eq!(vars.len(), self.0.var_count());

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

                Op::MinRegImmChoice { inout, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[inout].min_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegImmChoice { inout, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[inout].max_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegRegChoice { inout, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[inout].min_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegRegChoice { inout, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[inout].max_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
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
        choices: &mut Choices,
        data: &mut Self::Data,
    ) -> (f32, bool) {
        assert_eq!(vars.len(), self.0.var_count());
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

                Op::MinRegImmChoice { inout, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[inout], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegImmChoice { inout, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[inout], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegRegChoice { inout, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[inout], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegRegChoice { inout, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[inout], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[inout] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
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
        assert_eq!(vars.len(), self.0.var_count());
        assert_eq!(data.slots.len(), self.0.slot_count());

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
                Op::MinRegImmChoice { inout, imm, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].min(imm);
                    }
                }
                Op::MaxRegImmChoice { inout, imm, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].max(imm);
                    }
                }
                Op::MinRegRegChoice { inout, arg, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].min(v[arg][i]);
                    }
                }
                Op::MaxRegRegChoice { inout, arg, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].max(v[arg][i]);
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
        assert_eq!(vars.len(), self.0.var_count());
        assert_eq!(data.slots.len(), self.0.slot_count());

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
                Op::MinRegImmChoice { inout, imm, .. } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[inout][i] = v[inout][i].min(imm);
                    }
                }
                Op::MaxRegImmChoice { inout, imm, .. } => {
                    let imm: Grad = imm.into();
                    for i in 0..size {
                        v[inout][i] = v[inout][i].max(imm);
                    }
                }
                Op::MinRegRegChoice { inout, arg, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].min(v[arg][i]);
                    }
                }
                Op::MaxRegRegChoice { inout, arg, .. } => {
                    for i in 0..size {
                        v[inout][i] = v[inout][i].min(v[arg][i]);
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
fn min_choice(prev: f32, new: f32) -> (f32, Choice) {
    if prev < new {
        (prev, Choice::PrevValue)
    } else if new < prev {
        (new, Choice::NewValue)
    } else if prev.is_nan() || new.is_nan() {
        (std::f32::NAN, Choice::BothValues)
    } else {
        (prev, Choice::BothValues)
    }
}

/// Calculate the `max` value and a `Choice`, given two floating-point values
///
/// Unlike `f32::max`, this value will be `NAN` if _either_ input is `NAN`
fn max_choice(prev: f32, new: f32) -> (f32, Choice) {
    if prev > new {
        (prev, Choice::PrevValue)
    } else if new > prev {
        (new, Choice::NewValue)
    } else if prev.is_nan() || new.is_nan() {
        (std::f32::NAN, Choice::BothValues)
    } else {
        (prev, Choice::BothValues)
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
