use crate::{
    eval::{
        bulk::{BulkEvaluator, BulkEvaluatorData},
        tracing::{TracingEvaluator, TracingEvaluatorData},
        types::{Choice, Grad, Interval},
        EvaluatorStorage, Family,
    },
    vm::{ChoiceTape, Choices, Op, Tape, TapeData},
};

////////////////////////////////////////////////////////////////////////////////

/// Family of evaluators that use a local interpreter
#[derive(Clone)]
pub enum Eval {}

pub struct VmEvalGroup {
    pub chunk: Vec<Op>,
}

impl Family for Eval {
    /// This is interpreted, so we can use the maximum number of registers
    const REG_LIMIT: u8 = u8::MAX;

    type IntervalEval = AsmEval;
    type PointEval = AsmEval;
    type FloatSliceEval = AsmEval;
    type GradSliceEval = AsmEval;

    type TapeData = ();
    type GroupMetadata = VmEvalGroup;

    fn tile_sizes_3d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    fn tile_sizes_2d() -> &'static [usize] {
        &[256, 128, 64, 32, 16, 8]
    }

    fn build(
        tapes: &[ChoiceTape],
    ) -> (Self::TapeData, Vec<Self::GroupMetadata>) {
        let mut out = vec![];
        for t in tapes.iter() {
            let t = VmEvalGroup {
                chunk: t.tape.iter().rev().cloned().collect(),
            };
            out.push(t);
        }
        ((), out)
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
pub struct AsmEval {
    /// Flattened tape, in evaluation order
    active: Vec<Op>,

    /// Number of variables in the tape (used for assertions)
    var_count: usize,

    /// Number of slots in the tape (used for assertions)
    slot_count: usize,
}

impl AsmEval {
    fn iter_asm<'a>(&'a self) -> impl Iterator<Item = Op> + 'a {
        self.active.iter().cloned()
    }
}

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
    }
}

impl EvaluatorStorage<Eval> for AsmEval {
    type Storage = Vec<Op>;
    fn new_with_storage(tape: &Tape<Eval>, mut storage: Self::Storage) -> Self {
        storage.clear();
        for i in tape.active_groups().iter().rev() {
            storage.extend(tape.data().groups[*i].data.chunk.iter().cloned());
        }

        Self {
            active: storage,
            slot_count: tape.slot_count(),
            var_count: tape.var_count(),
        }
    }
    fn take(self) -> Option<Self::Storage> {
        Some(self.active)
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
        assert_eq!(vars.len(), self.var_count);

        let mut v = SlotArray(&mut data.slots);
        for op in self.iter_asm() {
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

                Op::CopyImmRegChoice { out, imm, choice } => {
                    v[out] = imm.into();
                    choices.set(choice);
                }

                Op::CopyImmMemChoice { out, imm, choice } => {
                    v[out] = imm.into();
                    choices.set(choice);
                }

                Op::CopyRegRegChoice { out, arg, choice } => {
                    v[out] = v[arg];
                    choices.set(choice);
                }

                Op::CopyRegMemChoice { out, arg, choice } => {
                    v[out] = v[arg];
                    choices.set(choice);
                }

                Op::MinMemImmChoice { mem, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[mem].min_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegImmChoice { reg, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[reg].min_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxMemImmChoice { mem, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[mem].max_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegImmChoice { reg, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[reg].max_choice(imm.into())
                    } else {
                        (imm.into(), Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinMemRegChoice { mem, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[mem].min_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegRegChoice { reg, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[reg].min_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxMemRegChoice { mem, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[mem].max_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegRegChoice { reg, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        v[reg].max_choice(v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[reg] = value;
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
                Op::CopyReg { out, arg } => {
                    v[out] = v[arg];
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
        assert_eq!(vars.len(), self.var_count);
        let mut simplify = false;
        let mut v = SlotArray(&mut data.slots);
        for op in self.iter_asm() {
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

                Op::CopyImmRegChoice { out, imm, choice } => {
                    v[out] = imm.into();
                    choices.set(choice);
                }

                Op::CopyImmMemChoice { out, imm, choice } => {
                    v[out] = imm.into();
                    choices.set(choice);
                }

                Op::CopyRegRegChoice { out, arg, choice } => {
                    v[out] = v[arg];
                    choices.set(choice);
                }

                Op::CopyRegMemChoice { out, arg, choice } => {
                    v[out] = v[arg];
                    choices.set(choice);
                }

                Op::MinMemImmChoice { mem, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[mem], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegImmChoice { reg, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[reg], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxMemImmChoice { mem, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[mem], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegImmChoice { reg, imm, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[reg], imm)
                    } else {
                        (imm, Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinMemRegChoice { mem, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[mem], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MinRegRegChoice { reg, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        min_choice(v[reg], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[reg] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxMemRegChoice { mem, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[mem], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[mem] = value;
                    match c {
                        Choice::PrevValue => (),
                        Choice::BothValues => choices.set(choice),
                        Choice::NewValue => choices.set_exclusive(choice),
                    }
                    simplify |= c != Choice::BothValues;
                }

                Op::MaxRegRegChoice { reg, arg, choice } => {
                    let (value, c) = if choices.has_value(choice) {
                        max_choice(v[reg], v[arg])
                    } else {
                        (v[arg], Choice::BothValues)
                    };
                    v[reg] = value;
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
                Op::CopyReg { out, arg } => {
                    v[out] = v[arg];
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
    T: From<f32> + Copy + Clone,
{
    fn prepare(&mut self, tape: &TapeData<Eval>, size: usize) {
        assert!(tape.reg_limit() == u8::MAX);
        let v: T = std::f32::NAN.into();
        self.slots.resize_with(tape.slot_count(), || {
            vec![v; size.max(self.slice_size)]
        });
        if size > self.slice_size {
            for s in self.slots.iter_mut() {
                s.resize(size, v);
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
        choices: &mut Choices,
        data: &mut Self::Data,
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.var_count);
        assert_eq!(data.slots.len(), self.slot_count);

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.iter_asm() {
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
                Op::CopyReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
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

                Op::CopyImmRegChoice { out, imm, choice } => {
                    for i in 0..size {
                        v[out][i] = imm.into();
                    }
                    choices.set(choice);
                }

                Op::CopyImmMemChoice { out, imm, choice } => {
                    for i in 0..size {
                        v[out][i] = imm.into();
                    }
                    choices.set(choice);
                }

                Op::CopyRegRegChoice { out, arg, choice } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                    choices.set(choice);
                }

                Op::CopyRegMemChoice { out, arg, choice } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                    choices.set(choice);
                }

                Op::MinMemImmChoice { mem, imm, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].min(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MinRegImmChoice { reg, imm, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].min(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxMemImmChoice { mem, imm, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].max(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxRegImmChoice { reg, imm, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].max(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MinMemRegChoice { mem, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].min(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MinRegRegChoice { reg, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].min(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxMemRegChoice { mem, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].max(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxRegRegChoice { reg, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].max(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
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
        choices: &mut Choices,
        data: &mut Self::Data,
    ) {
        assert_eq!(xs.len(), ys.len());
        assert_eq!(ys.len(), zs.len());
        assert_eq!(zs.len(), out.len());
        assert_eq!(vars.len(), self.var_count);
        assert_eq!(data.slots.len(), self.slot_count);

        let size = xs.len();
        assert!(data.slice_size >= size);

        let mut v = SlotArray(&mut data.slots);
        for op in self.iter_asm() {
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

                Op::CopyImmRegChoice { out, imm, choice } => {
                    for i in 0..size {
                        v[out][i] = imm.into();
                    }
                    choices.set(choice);
                }

                Op::CopyImmMemChoice { out, imm, choice } => {
                    for i in 0..size {
                        v[out][i] = imm.into();
                    }
                    choices.set(choice);
                }

                Op::CopyRegRegChoice { out, arg, choice } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                    choices.set(choice);
                }

                Op::CopyRegMemChoice { out, arg, choice } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
                    }
                    choices.set(choice);
                }

                Op::MinMemImmChoice { mem, imm, choice } => {
                    let imm: Grad = imm.into();
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].min(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MinRegImmChoice { reg, imm, choice } => {
                    let imm: Grad = imm.into();
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].min(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxMemImmChoice { mem, imm, choice } => {
                    let imm: Grad = imm.into();
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].max(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxRegImmChoice { reg, imm, choice } => {
                    let imm: Grad = imm.into();
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].max(imm);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = imm;
                        }
                    }
                    choices.set(choice);
                }
                Op::MinMemRegChoice { mem, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].min(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MinRegRegChoice { reg, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].min(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxMemRegChoice { mem, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[mem][i] = v[mem][i].max(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[mem][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
                }
                Op::MaxRegRegChoice { reg, arg, choice } => {
                    if choices.has_value(choice) {
                        for i in 0..size {
                            v[reg][i] = v[reg][i].max(v[arg][i]);
                        }
                    } else {
                        for i in 0..size {
                            v[reg][i] = v[arg][i];
                        }
                    }
                    choices.set(choice);
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
                Op::CopyReg { out, arg } => {
                    for i in 0..size {
                        v[out][i] = v[arg][i];
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
        out[0..size].copy_from_slice(&data.slots[0][0..size])
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
