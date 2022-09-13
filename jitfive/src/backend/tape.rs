use crate::{
    backend::{
        alloc::RegisterAllocator,
        asm::{AsmEval, AsmOp},
        common::{Choice, Simplify},
    },
    context::{Context, Node},
    op::{BinaryOpcode, Op, UnaryOpcode},
};

use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug)]
pub enum TapeOp {
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

/// `Tape` stores a pair of flat expressions suitable for evaluation:
/// - `ssa` is suitable for use during tape simplification
/// - `asm` is ready to be fed into an assembler, e.g. `dynasm`
///
/// We keep both because SSA form makes tape shortening easier, while the `asm`
/// data already has registers assigned.
pub struct Tape {
    pub ssa: SsaTape,
    pub asm: Vec<AsmOp>,
    reg_limit: u8,
}

impl Tape {
    pub fn from_ssa(ssa: SsaTape, reg_limit: u8) -> Self {
        let dummy = vec![Choice::Both; ssa.choice_count];
        let (ssa, asm) = ssa.simplify(&dummy, reg_limit);
        Self {
            ssa,
            asm,
            reg_limit,
        }
    }

    pub fn get_evaluator(&self) -> AsmEval {
        AsmEval::new(&self.asm)
    }
}

impl Simplify for Tape {
    fn simplify(&self, choices: &[Choice]) -> Self {
        let (ssa, asm) = self.ssa.simplify(choices, self.reg_limit);
        Self {
            ssa,
            asm,
            reg_limit: self.reg_limit,
        }
    }
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
pub struct SsaTape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<TapeOp>,

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

impl SsaTape {
    pub fn pretty_print(&self) {
        let mut data = self.data.iter().rev();
        let mut next = || *data.next().unwrap();
        for &op in self.tape.iter().rev() {
            match op {
                TapeOp::Input => {
                    let i = next();
                    let out = next();
                    println!("${out} = %{i}");
                }
                TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::CopyReg
                | TapeOp::SquareReg => {
                    let arg = next();
                    let out = next();
                    let op = match op {
                        TapeOp::NegReg => "NEG",
                        TapeOp::AbsReg => "ABS",
                        TapeOp::RecipReg => "RECIP",
                        TapeOp::SqrtReg => "SQRT",
                        TapeOp::SquareReg => "SQUARE",
                        TapeOp::CopyReg => "COPY",
                        _ => unreachable!(),
                    };
                    println!("${out} {op} ${arg}");
                }

                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::SubRegReg
                | TapeOp::MinRegReg
                | TapeOp::MaxRegReg => {
                    let rhs = next();
                    let lhs = next();
                    let out = next();
                    let op = match op {
                        TapeOp::AddRegReg => "ADD",
                        TapeOp::MulRegReg => "MUL",
                        TapeOp::SubRegReg => "SUB",
                        TapeOp::MinRegReg => "MIN",
                        TapeOp::MaxRegReg => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::SubImmReg
                | TapeOp::SubRegImm
                | TapeOp::MinRegImm
                | TapeOp::MaxRegImm => {
                    let imm = f32::from_bits(next());
                    let arg = next();
                    let out = next();
                    let (op, swap) = match op {
                        TapeOp::AddRegImm => ("ADD", false),
                        TapeOp::MulRegImm => ("MUL", false),
                        TapeOp::SubImmReg => ("SUB", true),
                        TapeOp::SubRegImm => ("SUB", false),
                        TapeOp::MinRegImm => ("MIN", false),
                        TapeOp::MaxRegImm => ("MAX", false),
                        _ => unreachable!(),
                    };
                    if swap {
                        println!("${out} = {op} {imm} ${arg}");
                    } else {
                        println!("${out} = {op} ${arg} {imm}");
                    }
                }
                TapeOp::CopyImm => {
                    let imm = f32::from_bits(next());
                    let out = next();
                    println!("${out} = COPY {imm}");
                }
            }
        }
    }

    pub fn simplify(
        &self,
        choices: &[Choice],
        reg_limit: u8,
    ) -> (Self, Vec<AsmOp>) {
        // If a node is active (i.e. has been used as an input, as we walk the
        // tape in reverse order), then store its new slot assignment here.
        let mut active = vec![None; self.tape.len()];
        let mut count = 0..;
        let mut choice_count = 0;

        // At this point, we don't know how long the shortened tape will be, but
        // we can be sure that it's <= our current tape length.
        let mut alloc = RegisterAllocator::new(reg_limit, self.tape.len());

        // The tape is constructed so that the output slot is first
        active[self.data[0] as usize] = Some(count.next().unwrap());

        // Other iterators to consume various arrays in order
        let mut data = self.data.iter();
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = vec![];
        let mut data_out = vec![];

        for &op in self.tape.iter() {
            use TapeOp::*;
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
                    ops_out.push(op);

                    match op {
                        Input => {
                            alloc.op_input(new_index, i.try_into().unwrap())
                        }
                        CopyImm => {
                            alloc.op_copy_imm(new_index, f32::from_bits(i))
                        }
                        _ => unreachable!(),
                    }
                }
                NegReg | AbsReg | RecipReg | SqrtReg | SquareReg => {
                    let arg = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(arg);
                    ops_out.push(op);

                    alloc.op_reg(new_index, arg, op);
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

                            alloc.op_reg(new_index, new_src, CopyReg);
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

                                alloc.op_reg(new_index, new_arg, CopyReg);
                            }
                            None => {
                                active[arg as usize] = Some(new_index);
                            }
                        },
                        Choice::Right => {
                            data_out.push(new_index);
                            data_out.push(imm);
                            ops_out.push(CopyImm);

                            alloc.op_copy_imm(new_index, f32::from_bits(imm));
                        }
                        Choice::Both => {
                            choice_count += 1;
                            let arg = *active[arg as usize]
                                .get_or_insert_with(|| count.next().unwrap());

                            data_out.push(new_index);
                            data_out.push(arg);
                            data_out.push(imm);
                            ops_out.push(op);

                            alloc.op_reg_imm(
                                new_index,
                                arg,
                                f32::from_bits(imm),
                                op,
                            );
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

                                alloc.op_reg(new_index, new_lhs, CopyReg);
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

                                alloc.op_reg(new_index, new_rhs, CopyReg);
                            }
                            None => {
                                active[rhs as usize] = Some(new_index);
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

                            alloc.op_reg_reg(new_index, lhs, rhs, op);
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

                    alloc.op_reg_reg(new_index, lhs, rhs, op);
                }
                AddRegImm | MulRegImm | SubRegImm | SubImmReg => {
                    let arg = *active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let imm = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(arg);
                    data_out.push(imm);
                    ops_out.push(op);

                    alloc.op_reg_imm(new_index, arg, f32::from_bits(imm), op);
                }
            }
        }

        assert_eq!(count.next().unwrap() as usize, ops_out.len());
        let alloc = alloc.take();
        assert!(ops_out.len() <= alloc.len());

        (
            SsaTape {
                tape: ops_out,
                data: data_out,
                choice_count,
            },
            alloc,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct SsaTapeBuilder {
    tape: Vec<TapeOp>,
    data: Vec<u32>,

    mapping: BTreeMap<Node, u32>,
    constants: BTreeMap<Node, f32>,
    choice_count: usize,
}

#[derive(Debug)]
enum Location {
    Slot(u32),
    Immediate(f32),
}

impl SsaTapeBuilder {
    pub fn new() -> Self {
        Self {
            tape: vec![],
            data: vec![],
            mapping: BTreeMap::new(),
            constants: BTreeMap::new(),
            choice_count: 0,
        }
    }

    pub fn finish(self) -> SsaTape {
        SsaTape {
            tape: self.tape,
            data: self.data,
            choice_count: self.choice_count,
        }
    }

    fn get_allocated_value(&mut self, node: Node) -> Location {
        if let Some(r) = self.mapping.get(&node).cloned() {
            Location::Slot(r)
        } else {
            let c = self.constants.get(&node).unwrap();
            Location::Immediate(*c)
        }
    }

    /// Ensure that the given node is mapped.
    ///
    /// This must be called before `step` uses the node (as either parent or
    /// child).
    pub fn declare_node(&mut self, node: Node, op: Op) {
        match op {
            Op::Const(c) => {
                self.constants.insert(node, c.0 as f32);
            }
            _ => {
                let index: u32 = self.mapping.len().try_into().unwrap();
                self.mapping.entry(node).or_insert(index);
            }
        }
    }

    pub fn step(&mut self, node: Node, op: Op, ctx: &Context) {
        let index = self.mapping.get(&node).cloned();
        let op = match op {
            Op::Var(v) => {
                let arg = match ctx.get_var_by_index(v).unwrap() {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    i => panic!("Unexpected input index: {i}"),
                };
                self.data.push(index.unwrap());
                self.data.push(arg);
                Some(TapeOp::Input)
            }
            Op::Const(c) => {
                // Skip this (because it's not inserted into the tape),
                // recording its value for use as an immediate later.
                self.constants.insert(node, c.0 as f32);
                assert!(index.is_none());
                None
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.get_allocated_value(lhs);
                let rhs = self.get_allocated_value(rhs);
                let index = index.unwrap();

                let f = match op {
                    BinaryOpcode::Add => (
                        TapeOp::AddRegReg,
                        TapeOp::AddRegImm,
                        TapeOp::AddRegImm,
                    ),
                    BinaryOpcode::Mul => (
                        TapeOp::MulRegReg,
                        TapeOp::MulRegImm,
                        TapeOp::MulRegImm,
                    ),
                    BinaryOpcode::Sub => (
                        TapeOp::SubRegReg,
                        TapeOp::SubRegImm,
                        TapeOp::SubImmReg,
                    ),
                    BinaryOpcode::Min => (
                        TapeOp::MinRegReg,
                        TapeOp::MinRegImm,
                        TapeOp::MinRegImm,
                    ),
                    BinaryOpcode::Max => (
                        TapeOp::MaxRegReg,
                        TapeOp::MaxRegImm,
                        TapeOp::MaxRegImm,
                    ),
                };

                if matches!(op, BinaryOpcode::Min | BinaryOpcode::Max) {
                    self.choice_count += 1;
                }

                let op = match (lhs, rhs) {
                    (Location::Slot(lhs), Location::Slot(rhs)) => {
                        self.data.push(index);
                        self.data.push(lhs);
                        self.data.push(rhs);
                        f.0
                    }
                    (Location::Slot(arg), Location::Immediate(imm)) => {
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                        f.1
                    }
                    (Location::Immediate(imm), Location::Slot(arg)) => {
                        self.data.push(index);
                        self.data.push(arg);
                        self.data.push(imm.to_bits());
                        f.2
                    }
                    (Location::Immediate(..), Location::Immediate(..)) => {
                        panic!("Cannot handle f(imm, imm)")
                    }
                };
                Some(op)
            }
            Op::Unary(op, lhs) => {
                let lhs = match self.get_allocated_value(lhs) {
                    Location::Slot(r) => r,
                    Location::Immediate(..) => {
                        panic!("Cannot handle f(imm)")
                    }
                };
                let index = index.unwrap();
                let op = match op {
                    UnaryOpcode::Neg => TapeOp::NegReg,
                    UnaryOpcode::Abs => TapeOp::AbsReg,
                    UnaryOpcode::Recip => TapeOp::RecipReg,
                    UnaryOpcode::Sqrt => TapeOp::SqrtReg,
                    UnaryOpcode::Square => TapeOp::SquareReg,
                };
                self.data.push(index);
                self.data.push(lhs);
                Some(op)
            }
        };

        if let Some(op) = op {
            self.tape.push(op);
        }
    }
}

impl Default for SsaTapeBuilder {
    fn default() -> Self {
        Self::new()
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
        let tape = ctx.get_tape(min, u8::MAX);
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

        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(1.0, 2.0, 0.0), 2.0);
        assert_eq!(eval.f(3.0, 2.0, 0.0), 2.0);

        let one = ctx.constant(1.0);
        let min = ctx.min(x, one).unwrap();
        let tape = ctx.get_tape(min, u8::MAX);
        let mut eval = tape.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);

        let t = tape.simplify(&[Choice::Left]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 3.0);

        let t = tape.simplify(&[Choice::Right]);
        let mut eval = t.get_evaluator();
        assert_eq!(eval.f(0.5, 0.0, 0.0), 1.0);
        assert_eq!(eval.f(3.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn test_ring() {
        let mut ctx = crate::context::Context::new();
        let c0 = ctx.constant(0.5);
        let x = ctx.x();
        let y = ctx.y();
        let x2 = ctx.square(x).unwrap();
        let y2 = ctx.square(y).unwrap();
        let r = ctx.add(x2, y2).unwrap();
        let c6 = ctx.sub(r, c0).unwrap();
        let c7 = ctx.constant(0.25);
        let c8 = ctx.sub(c7, r).unwrap();
        let c9 = ctx.max(c8, c6).unwrap();

        let tape = ctx.get_tape(c9, u8::MAX);
        assert_eq!(tape.ssa.tape.len(), 8);
    }
}
