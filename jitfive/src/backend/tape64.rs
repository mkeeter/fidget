use crate::{
    backend::{
        common::{Choice, NodeIndex, Op, VarIndex},
        dynasm::AsmOp,
    },
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    scheduled::Scheduled,
    util::indexed::IndexMap,
};

use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug)]
pub enum ClauseOp64 {
    /// Reads one of the inputs (X, Y, Z)
    Input,
    /// Copy an immediate to a register
    CopyImm,

    /// Negates a register
    NegReg,
    /// Takes the absolute value of a register
    AbsReg,
    /// Takes the reciprocal of a register
    RecipReg,
    /// Takes the square root of a register
    SqrtReg,
    /// Squares a register
    SquareReg,

    /// Copies the given register
    CopyReg,

    /// Add a register and an immediate
    AddRegImm,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Subtract a register from an immediate
    SubImmReg,
    /// Subtract an immediate from a register
    SubRegImm,

    /// Adds two registers
    AddRegReg,
    /// Multiplies two registers
    MulRegReg,
    /// Subtracts two registers
    SubRegReg,

    /// Compute the minimum of a register and an immediate
    MinRegImm,
    /// Compute the maximum of a register and an immediate
    MaxRegImm,
    /// Compute the minimum of two registers
    MinRegReg,
    /// Compute the maximum of two registers
    MaxRegReg,
}

/// Tape storing... stuff
/// - 4-byte opcode
/// - 4-byte output register
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs, arguments, and immediates are packed into the `data` array
///
/// All slot addressing is absolute.
#[derive(Clone, Debug)]
pub struct Tape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<ClauseOp64>,

    /// Variable-length data for tape clauses.
    ///
    /// Data is densely packed in the order
    /// - output slot
    /// - lhs slot (or input)
    /// - rhs slot (or immediate)
    ///
    /// i.e. a unary operation would only store two items in this array
    pub data: Vec<u32>,

    /// Number of choice operations in the tape
    pub choice_count: usize,
}

impl Tape {
    pub fn new(t: &Scheduled) -> Self {
        let mut builder = TapeBuilder::new(t);
        builder.run();
        builder.tape.reverse();
        builder.data.reverse();
        Self {
            tape: builder.tape,
            data: builder.data,
            choice_count: builder.choice_count,
        }
    }

    /// Builds an evaluator which takes a (read-only) reference to this tape
    pub fn get_evaluator(&self) -> TapeEval {
        TapeEval {
            tape: self,
            slots: vec![0.0; self.tape.len()],
        }
    }

    pub fn pretty_print(&self) {
        let mut data = self.data.iter().rev();
        let mut next = || *data.next().unwrap();
        for &op in self.tape.iter().rev() {
            match op {
                ClauseOp64::Input => {
                    let i = next();
                    let out = next();
                    println!("${out} = %{i}");
                }
                ClauseOp64::NegReg
                | ClauseOp64::AbsReg
                | ClauseOp64::RecipReg
                | ClauseOp64::SqrtReg
                | ClauseOp64::CopyReg
                | ClauseOp64::SquareReg => {
                    let arg = next();
                    let out = next();
                    let op = match op {
                        ClauseOp64::NegReg => "NEG",
                        ClauseOp64::AbsReg => "ABS",
                        ClauseOp64::RecipReg => "RECIP",
                        ClauseOp64::SqrtReg => "SQRT",
                        ClauseOp64::SquareReg => "SQUARE",
                        ClauseOp64::CopyReg => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} {op} ${arg}");
                }

                ClauseOp64::AddRegReg
                | ClauseOp64::MulRegReg
                | ClauseOp64::SubRegReg
                | ClauseOp64::MinRegReg
                | ClauseOp64::MaxRegReg => {
                    let rhs = next();
                    let lhs = next();
                    let out = next();
                    let op = match op {
                        ClauseOp64::AddRegReg => "ADD",
                        ClauseOp64::MulRegReg => "MUL",
                        ClauseOp64::SubRegReg => "SUB",
                        ClauseOp64::MinRegReg => "MIN",
                        ClauseOp64::MaxRegReg => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                ClauseOp64::AddRegImm
                | ClauseOp64::MulRegImm
                | ClauseOp64::SubImmReg
                | ClauseOp64::SubRegImm
                | ClauseOp64::MinRegImm
                | ClauseOp64::MaxRegImm => {
                    let imm = f32::from_bits(next());
                    let arg = next();
                    let out = next();
                    let (op, swap) = match op {
                        ClauseOp64::AddRegImm => ("ADD", false),
                        ClauseOp64::MulRegImm => ("MUL", false),
                        ClauseOp64::SubImmReg => ("SUB", true),
                        ClauseOp64::SubRegImm => ("SUB", false),
                        ClauseOp64::MinRegImm => ("MIN", false),
                        ClauseOp64::MaxRegImm => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} ${arg} {imm}");
                    } else {
                        println!("${out} = {op} {imm} ${arg}");
                    }
                }
                ClauseOp64::CopyImm => {
                    let imm = f32::from_bits(next());
                    let out = next();
                    println!("${out} = COPY {imm}");
                }
            }
        }
    }

    pub fn simplify(&self, choices: &[Choice]) -> (Self, Vec<AsmOp>) {
        // If a node is active (i.e. has been used as an input, as we walk the
        // tape in reverse order), then store its new slot assignment here.
        let mut active = vec![None; self.tape.len()];
        let mut count = 0..;
        let mut choice_count = 0;

        // The tape is constructed so that the output slot is first
        active[self.data[0] as usize] = Some(count.next().unwrap());

        // Other iterators to consume various arrays in order
        let mut data = self.data.iter();
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = vec![];
        let mut data_out = vec![];

        for &op in self.tape.iter() {
            use ClauseOp64::*;
            let index = *data.next().unwrap();
            if active[index as usize].is_none() {
                match op {
                    Input | CopyImm | NegReg | AbsReg | RecipReg | SqrtReg
                    | SquareReg | CopyReg => {
                        data.next().unwrap();
                    }
                    AddRegImm | MulRegImm | SubRegImm | SubImmReg
                    | AddRegReg | MulRegReg | SubRegReg => {
                        data.next().unwrap();
                        data.next().unwrap();
                    }

                    MinRegImm | MaxRegImm | MinRegReg | MaxRegReg => {
                        data.next().unwrap();
                        data.next().unwrap();
                        choice_iter.next().unwrap();
                    }
                }
                continue;
            }

            // Because we reassign nodes when they're used as an *input*
            // (while walking the tape in reverse), this node must have been
            // assigned already.
            let new_index = active[index as usize].unwrap();

            match op {
                Input | CopyImm => {
                    let i = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(i);
                    ops_out.push(op)
                }
                NegReg | AbsReg | RecipReg | SqrtReg | SquareReg => {
                    let arg = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(arg);
                    ops_out.push(op);
                }
                CopyReg => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    let src = *data.next().unwrap();
                    match active[src as usize] {
                        Some(new_src) => {
                            data_out.push(new_index);
                            data_out.push(new_src);
                            ops_out.push(op);
                        }
                        None => {
                            active[src as usize] = Some(new_index);
                        }
                    }
                }
                MinRegImm | MaxRegImm => {
                    let arg = *data.next().unwrap();
                    let imm = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match active[arg as usize] {
                            Some(new_arg) => {
                                data_out.push(new_index);
                                data_out.push(new_arg);
                                ops_out.push(CopyReg);
                            }
                            None => {
                                active[arg as usize] = Some(new_index);
                            }
                        },
                        Choice::Right => {
                            data_out.push(new_index);
                            data_out.push(imm);
                            ops_out.push(CopyImm);
                        }
                        Choice::Both => {
                            choice_count += 1;
                            let arg = *active[arg as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            data_out.push(new_index);
                            data_out.push(arg);
                            data_out.push(imm);
                            ops_out.push(op);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                MinRegReg | MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match active[lhs as usize] {
                            Some(new_lhs) => {
                                data_out.push(new_index);
                                data_out.push(new_lhs);
                                ops_out.push(CopyReg);
                            }
                            None => {
                                active[lhs as usize] = Some(new_index);
                            }
                        },
                        Choice::Right => match active[rhs as usize] {
                            Some(new_rhs) => {
                                data_out.push(new_index);
                                data_out.push(new_rhs);
                                ops_out.push(CopyReg);
                            }
                            None => {
                                active[lhs as usize] = Some(new_index);
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            let lhs = *active[lhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            let rhs = *active[rhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            data_out.push(new_index);
                            data_out.push(lhs);
                            data_out.push(rhs);
                            ops_out.push(op);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                AddRegReg | MulRegReg | SubRegReg => {
                    let lhs = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let rhs = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(lhs);
                    data_out.push(rhs);
                    ops_out.push(op);
                }
                AddRegImm | MulRegImm | SubRegImm | SubImmReg => {
                    let arg = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let imm = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(arg);
                    data_out.push(imm);
                    ops_out.push(op);
                }
            }
        }
        assert_eq!(count.next().unwrap() as usize, ops_out.len());
        (
            Tape {
                tape: ops_out,
                data: data_out,
                choice_count,
            },
            vec![],
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

struct TapeBuilder<'a> {
    iter: std::slice::Iter<'a, (NodeIndex, Op)>,

    tape: Vec<ClauseOp64>,
    data: Vec<u32>,

    vars: &'a IndexMap<String, VarIndex>,
    mapping: BTreeMap<NodeIndex, u32>,
    constants: BTreeMap<NodeIndex, f32>,
    choice_count: usize,
}

enum Location {
    Slot(u32),
    Immediate(f32),
}

impl<'a> TapeBuilder<'a> {
    fn new(t: &'a Scheduled) -> Self {
        Self {
            iter: t.tape.iter(),
            tape: vec![],
            data: vec![],
            vars: &t.vars,
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
        }
    }

    fn get_allocated_value(&mut self, node: NodeIndex) -> Location {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Location::Slot(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Location::Immediate(*c)
        }
    }

    fn run(&mut self) {
        while let Some(&(n, op)) = self.iter.next() {
            self.step(n, op);
        }
    }

    fn step(&mut self, node: NodeIndex, op: Op) {
        let index = self.mapping.len().try_into().unwrap();
        match op {
            Op::Var(v) => {
                let arg = match self.vars.get_by_index(v).unwrap().as_str() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    _ => panic!(),
                };
                self.tape.push(ClauseOp64::Input);
                self.data.push(arg);
                self.data.push(index);
            }
            Op::Const(c) => {
                // Skip this (because it's not inserted into the tape),
                // recording its value for use as an immediate later.
                self.constants.insert(node, c as f32);
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);

                let f = match op {
                    BinaryOpcode::Add => (
                        ClauseOp64::AddRegReg,
                        ClauseOp64::AddRegImm,
                        ClauseOp64::AddRegImm,
                    ),
                    BinaryOpcode::Mul => (
                        ClauseOp64::MulRegReg,
                        ClauseOp64::MulRegImm,
                        ClauseOp64::MulRegImm,
                    ),
                    BinaryOpcode::Sub => (
                        ClauseOp64::SubRegReg,
                        ClauseOp64::SubRegImm,
                        ClauseOp64::SubImmReg,
                    ),
                };

                match (lhs, rhs) {
                    (Location::Slot(lhs), Location::Slot(rhs)) => {
                        self.tape.push(f.0);
                        self.data.push(rhs);
                        self.data.push(lhs);
                        self.data.push(index);
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        self.tape.push(f.1);
                        self.data.push(imm.to_bits());
                        self.data.push(arg);
                        self.data.push(index);
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        self.tape.push(f.2);
                        self.data.push(imm.to_bits());
                        self.data.push(arg);
                        self.data.push(index);
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                }
            }
            Op::BinaryChoice(op, lhs, rhs, ..) => {
                self.choice_count += 1;
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);

                let f = match op {
                    BinaryChoiceOpcode::Min => {
                        (ClauseOp64::MinRegReg, ClauseOp64::MinRegImm)
                    }
                    BinaryChoiceOpcode::Max => {
                        (ClauseOp64::MaxRegReg, ClauseOp64::MaxRegImm)
                    }
                };

                match (lhs, rhs) {
                    (Location::Slot(lhs), Location::Slot(rhs)) => {
                        self.tape.push(f.0);
                        self.data.push(rhs);
                        self.data.push(lhs);
                        self.data.push(index);
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        self.tape.push(f.1);
                        self.data.push(imm.to_bits());
                        self.data.push(arg);
                        self.data.push(index);
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        self.tape.push(f.1);
                        self.data.push(imm.to_bits());
                        self.data.push(arg);
                        self.data.push(index);
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                }
            }
            Op::Unary(op, lhs) => {
                let lhs = match self.get_allocated_value(lhs) {
                    Location::Slot(r) => r,
                    Location::Immediate(..) => {
                        panic!("Cannot handle f(imm)")
                    }
                };
                let op = match op {
                    UnaryOpcode::Neg => ClauseOp64::NegReg,
                    UnaryOpcode::Abs => ClauseOp64::AbsReg,
                    UnaryOpcode::Recip => ClauseOp64::RecipReg,
                    UnaryOpcode::Sqrt => ClauseOp64::SqrtReg,
                    UnaryOpcode::Square => ClauseOp64::SquareReg,
                };
                self.tape.push(op);
                self.data.push(lhs);
                self.data.push(index);
            }
        };
        let r = self.mapping.insert(node, index);
        assert!(r.is_none());
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Workspace to evaluate a tape
pub struct TapeEval<'a> {
    tape: &'a Tape,
    slots: Vec<f32>,
}

impl<'a> TapeEval<'a> {
    fn v(&self, i: u32) -> f32 {
        self.slots[i as usize]
    }
    pub fn f(&mut self, x: f32, y: f32, z: f32) -> f32 {
        let mut data = self.tape.data.iter().rev();
        let mut next = || *data.next().unwrap();
        for &op in self.tape.tape.iter().rev() {
            let out = match op {
                ClauseOp64::Input => match next() {
                    0 => x,
                    1 => y,
                    2 => z,
                    _ => panic!(),
                },
                ClauseOp64::NegReg
                | ClauseOp64::AbsReg
                | ClauseOp64::RecipReg
                | ClauseOp64::SqrtReg
                | ClauseOp64::CopyReg
                | ClauseOp64::SquareReg => {
                    let arg = self.v(next());
                    match op {
                        ClauseOp64::NegReg => -arg,
                        ClauseOp64::AbsReg => arg.abs(),
                        ClauseOp64::RecipReg => 1.0 / arg,
                        ClauseOp64::SqrtReg => arg.sqrt(),
                        ClauseOp64::SquareReg => arg * arg,
                        ClauseOp64::CopyReg => arg,
                        _ => unreachable!(),
                    }
                }

                ClauseOp64::AddRegReg
                | ClauseOp64::MulRegReg
                | ClauseOp64::SubRegReg
                | ClauseOp64::MinRegReg
                | ClauseOp64::MaxRegReg => {
                    let rhs = self.v(next());
                    let lhs = self.v(next());
                    match op {
                        ClauseOp64::AddRegReg => lhs + rhs,
                        ClauseOp64::MulRegReg => lhs * rhs,
                        ClauseOp64::SubRegReg => lhs - rhs,
                        ClauseOp64::MinRegReg => lhs.min(rhs),
                        ClauseOp64::MaxRegReg => lhs.max(rhs),
                        _ => unreachable!(),
                    }
                }

                ClauseOp64::AddRegImm
                | ClauseOp64::MulRegImm
                | ClauseOp64::SubImmReg
                | ClauseOp64::SubRegImm
                | ClauseOp64::MinRegImm
                | ClauseOp64::MaxRegImm => {
                    let imm = f32::from_bits(next());
                    let arg = self.v(next());
                    match op {
                        ClauseOp64::AddRegImm => arg + imm,
                        ClauseOp64::MulRegImm => arg * imm,
                        ClauseOp64::SubImmReg => imm - arg,
                        ClauseOp64::SubRegImm => arg - imm,
                        ClauseOp64::MinRegImm => arg.min(imm),
                        ClauseOp64::MaxRegImm => arg.max(imm),
                        _ => unreachable!(),
                    }
                }
                ClauseOp64::CopyImm => f32::from_bits(next()),
            };
            self.slots[next() as usize] = out;
        }
        self.slots[self.tape.data[0] as usize]
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::common::Choice;

    #[test]
    fn basic_interpreter() {
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let one = ctx.constant(1.0);
        let sum = ctx.add(x, one).unwrap();
        let min = ctx.min(sum, y).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(1.0, 3.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 3.5, 0.0), 3.5);
    }

    #[test]
    fn test_push() {
        let mut ctx = crate::context::Context::new();
        let x = ctx.x();
        let y = ctx.y();
        let min = ctx.min(x, y).unwrap();

        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.0.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.0.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let scheduled = crate::scheduled::schedule(&ctx, min);
        let tape = Tape::new(&scheduled);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.0.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.0.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);
    }
}
