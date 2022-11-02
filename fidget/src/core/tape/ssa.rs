use crate::{
    asm::{AsmTape, RegisterAllocator},
    eval::Choice,
    tape::TapeOp,
};

/// Instruction tape, storing [`TapeOp`](crate::tape::TapeOp) in SSA form
///
/// Each operation has the following parameters
/// - 4-byte opcode (required)
/// - 4-byte output register (required)
/// - 4-byte LHS register
/// - 4-byte RHS register (or immediate `f32`)
///
/// Outputs, arguments, and immediates are packed into the `data` array
///
/// All register addressing is absolute.
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
    /// Returns the number of opcodes in the tape
    pub fn len(&self) -> usize {
        self.tape.len()
    }
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
                    println!("${out} = {op} ${arg}");
                }

                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::DivRegReg
                | TapeOp::SubRegReg
                | TapeOp::MinRegReg
                | TapeOp::MaxRegReg => {
                    let rhs = next();
                    let lhs = next();
                    let out = next();
                    let op = match op {
                        TapeOp::AddRegReg => "ADD",
                        TapeOp::MulRegReg => "MUL",
                        TapeOp::DivRegReg => "DIV",
                        TapeOp::SubRegReg => "SUB",
                        TapeOp::MinRegReg => "MIN",
                        TapeOp::MaxRegReg => "MAX",
                        _ => unreachable!(),
                    };
                    println!("${out} = {op} ${lhs} ${rhs}");
                }

                TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg
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
                        TapeOp::DivImmReg => ("DIV", true),
                        TapeOp::DivRegImm => ("DIV", false),
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

    /// Lowers the tape to assembly with a particular register limit
    ///
    /// Note that if you _also_ want to simplify the tape, it's more efficient
    /// to use [`simplify`](Self::simplify), which simultaneously simplifies
    /// **and** performs register allocation in a single pass.
    pub fn get_asm(&self, reg_limit: u8) -> AsmTape {
        let mut alloc = RegisterAllocator::new(reg_limit, self.tape.len());
        let mut data = self.data.iter();
        for &op in self.tape.iter() {
            let index = *data.next().unwrap();

            match op {
                TapeOp::Input => {
                    let i = *data.next().unwrap();
                    alloc.op_input(index, i.try_into().unwrap());
                }
                TapeOp::CopyImm => {
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_copy_imm(index, imm);
                }
                TapeOp::CopyReg
                | TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::SquareReg => {
                    let arg = *data.next().unwrap();
                    alloc.op_reg(index, arg, op);
                }
                TapeOp::MinRegImm
                | TapeOp::MaxRegImm
                | TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg
                | TapeOp::SubRegImm
                | TapeOp::SubImmReg => {
                    let arg = *data.next().unwrap();
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_reg_imm(index, arg, imm, op);
                }
                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::DivRegReg
                | TapeOp::SubRegReg
                | TapeOp::MinRegReg
                | TapeOp::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    alloc.op_reg_reg(index, lhs, rhs, op);
                }
            }
        }
        alloc.take()
    }

    pub fn simplify_with(
        &self,
        choices: &[Choice],
        reg_limit: u8,
        workspace: &mut Workspace,
    ) -> (Self, AsmTape) {
        workspace.reset(reg_limit, self.tape.len());
        let mut count = 0..;
        let mut choice_count = 0;

        // The tape is constructed so that the output slot is first
        workspace.active[self.data[0] as usize] = Some(count.next().unwrap());

        // Other iterators to consume various arrays in order
        let mut data = self.data.iter();
        let mut choice_iter = choices.iter().rev();

        let mut ops_out = Vec::with_capacity(self.tape.len());
        let mut data_out = Vec::with_capacity(self.data.len());

        for &op in self.tape.iter() {
            let index = *data.next().unwrap();
            if workspace.active[index as usize].is_none() {
                for _ in 0..op.data_count() {
                    data.next().unwrap();
                }
                for _ in 0..op.choice_count() {
                    choice_iter.next().unwrap();
                }
                continue;
            }

            // Because we reassign nodes when they're used as an *input*
            // (while walking the tape in reverse), this node must have been
            // assigned already.
            let new_index = workspace.active[index as usize].unwrap();

            match op {
                TapeOp::Input | TapeOp::CopyImm => {
                    let i = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(i);
                    ops_out.push(op);

                    match op {
                        TapeOp::Input => workspace
                            .alloc
                            .op_input(new_index, i.try_into().unwrap()),
                        TapeOp::CopyImm => workspace
                            .alloc
                            .op_copy_imm(new_index, f32::from_bits(i)),
                        _ => unreachable!(),
                    }
                }
                TapeOp::NegReg
                | TapeOp::AbsReg
                | TapeOp::RecipReg
                | TapeOp::SqrtReg
                | TapeOp::SquareReg => {
                    let arg = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(arg);
                    ops_out.push(op);

                    workspace.alloc.op_reg(new_index, arg, op);
                }
                TapeOp::CopyReg => {
                    // CopyReg effectively does
                    //      dst <= src
                    // If src has not yet been used (as we iterate backwards
                    // through the tape), then we can replace it with dst
                    // everywhere!
                    let src = *data.next().unwrap();
                    match workspace.active[src as usize] {
                        Some(new_src) => {
                            data_out.push(new_index);
                            data_out.push(new_src);
                            ops_out.push(op);

                            workspace.alloc.op_reg(
                                new_index,
                                new_src,
                                TapeOp::CopyReg,
                            );
                        }
                        None => {
                            workspace.active[src as usize] = Some(new_index);
                        }
                    }
                }
                TapeOp::MinRegImm | TapeOp::MaxRegImm => {
                    let arg = *data.next().unwrap();
                    let imm = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[arg as usize] {
                            Some(new_arg) => {
                                data_out.push(new_index);
                                data_out.push(new_arg);
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_arg,
                                    TapeOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[arg as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Right => {
                            data_out.push(new_index);
                            data_out.push(imm);
                            ops_out.push(TapeOp::CopyImm);

                            workspace
                                .alloc
                                .op_copy_imm(new_index, f32::from_bits(imm));
                        }
                        Choice::Both => {
                            choice_count += 1;
                            let arg = *workspace.active[arg as usize]
                                .get_or_insert_with(|| count.next().unwrap());

                            data_out.push(new_index);
                            data_out.push(arg);
                            data_out.push(imm);
                            ops_out.push(op);

                            workspace.alloc.op_reg_imm(
                                new_index,
                                arg,
                                f32::from_bits(imm),
                                op,
                            );
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                TapeOp::MinRegReg | TapeOp::MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    match choice_iter.next().unwrap() {
                        Choice::Left => match workspace.active[lhs as usize] {
                            Some(new_lhs) => {
                                data_out.push(new_index);
                                data_out.push(new_lhs);
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_lhs,
                                    TapeOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[lhs as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Right => match workspace.active[rhs as usize] {
                            Some(new_rhs) => {
                                data_out.push(new_index);
                                data_out.push(new_rhs);
                                ops_out.push(TapeOp::CopyReg);

                                workspace.alloc.op_reg(
                                    new_index,
                                    new_rhs,
                                    TapeOp::CopyReg,
                                );
                            }
                            None => {
                                workspace.active[rhs as usize] =
                                    Some(new_index);
                            }
                        },
                        Choice::Both => {
                            choice_count += 1;
                            let lhs = *workspace.active[lhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            let rhs = *workspace.active[rhs as usize]
                                .get_or_insert_with(|| count.next().unwrap());
                            data_out.push(new_index);
                            data_out.push(lhs);
                            data_out.push(rhs);
                            ops_out.push(op);

                            workspace.alloc.op_reg_reg(new_index, lhs, rhs, op);
                        }
                        Choice::Unknown => panic!("oh no"),
                    }
                }
                TapeOp::AddRegReg
                | TapeOp::MulRegReg
                | TapeOp::SubRegReg
                | TapeOp::DivRegReg => {
                    let lhs = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let rhs = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    data_out.push(new_index);
                    data_out.push(lhs);
                    data_out.push(rhs);
                    ops_out.push(op);

                    workspace.alloc.op_reg_reg(new_index, lhs, rhs, op);
                }
                TapeOp::AddRegImm
                | TapeOp::MulRegImm
                | TapeOp::SubRegImm
                | TapeOp::SubImmReg
                | TapeOp::DivRegImm
                | TapeOp::DivImmReg => {
                    let arg = *workspace.active[*data.next().unwrap() as usize]
                        .get_or_insert_with(|| count.next().unwrap());
                    let imm = *data.next().unwrap();
                    data_out.push(new_index);
                    data_out.push(arg);
                    data_out.push(imm);
                    ops_out.push(op);

                    workspace.alloc.op_reg_imm(
                        new_index,
                        arg,
                        f32::from_bits(imm),
                        op,
                    );
                }
            }
        }

        assert_eq!(count.next().unwrap() as usize, ops_out.len());
        let asm_tape = workspace.alloc.take();
        assert!(ops_out.len() <= asm_tape.len());

        (
            SsaTape {
                tape: ops_out,
                data: data_out,
                choice_count,
            },
            asm_tape,
        )
    }

    pub fn simplify(
        &self,
        choices: &[Choice],
        reg_limit: u8,
    ) -> (Self, AsmTape) {
        self.simplify_with(choices, reg_limit, &mut Default::default())
    }
}

pub struct Workspace {
    alloc: RegisterAllocator,
    active: Vec<Option<u32>>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            alloc: RegisterAllocator::new(1, 1),
            active: vec![],
        }
    }
}

impl Workspace {
    fn reset(&mut self, num_registers: u8, tape_len: usize) {
        self.alloc.reset(num_registers, tape_len);
        self.active.fill(None);
        self.active.resize(tape_len, None);
    }
}
