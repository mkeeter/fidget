use crate::{
    asm::{AsmOp, RegisterAllocator},
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

    /// Lowers the tape to assembly with a particular register limit
    ///
    /// Note that if you _also_ want to simplify the tape, it's more efficient
    /// to use [`simplify`](Self::simplify), which simultaneously simplifies
    /// **and** performs register allocation in a single pass.
    pub fn get_asm(&self, reg_limit: u8) -> Vec<AsmOp> {
        let mut alloc = RegisterAllocator::new(reg_limit, self.tape.len());
        let mut data = self.data.iter();
        for &op in self.tape.iter() {
            use TapeOp::*;
            let index = *data.next().unwrap();

            match op {
                Input => {
                    let i = *data.next().unwrap();
                    alloc.op_input(index, i.try_into().unwrap());
                }
                CopyImm => {
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_copy_imm(index, imm);
                }
                CopyReg | NegReg | AbsReg | RecipReg | SqrtReg | SquareReg => {
                    let arg = *data.next().unwrap();
                    alloc.op_reg(index, arg, op);
                }
                MinRegImm | MaxRegImm | AddRegImm | MulRegImm | SubRegImm
                | SubImmReg => {
                    let arg = *data.next().unwrap();
                    let imm = f32::from_bits(*data.next().unwrap());
                    alloc.op_reg_imm(index, arg, imm, op);
                }
                AddRegReg | MulRegReg | SubRegReg | MinRegReg | MaxRegReg => {
                    let lhs = *data.next().unwrap();
                    let rhs = *data.next().unwrap();
                    alloc.op_reg_reg(index, lhs, rhs, op);
                }
            }
        }
        alloc.take()
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

        let mut ops_out = Vec::with_capacity(self.tape.len());
        let mut data_out = Vec::with_capacity(self.data.len());

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
