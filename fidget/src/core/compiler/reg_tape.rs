//! Tape used for evaluation
use crate::compiler::{RegOp, RegisterAllocator, SsaOp, SsaTape};

/// Low-level tape for use with the Fidget virtual machine (or to be lowered
/// further into machine instructions).
#[derive(Clone, Default)]
pub struct RegTape {
    tape: Vec<RegOp>,

    /// Total allocated slots, including registers and memory
    pub(super) slot_count: u32,
}

impl RegTape {
    /// Lowers the tape to assembly with a particular register limit
    ///
    /// Note that if you _also_ want to simplify the tape, it's more efficient
    /// to use [`VmData::simplify`](crate::vm::VmData::simplify), which
    /// simultaneously simplifies **and** performs register allocation in a
    /// single pass.
    pub fn new<const N: usize>(ssa: &SsaTape) -> Self {
        let mut alloc = RegisterAllocator::<N>::new(ssa.len());
        for &op in ssa.iter() {
            alloc.op(op)
        }
        alloc.finalize()
    }

    /// Builds a new empty tape, with one allocated slot
    pub(crate) fn empty() -> Self {
        Self {
            tape: vec![],
            slot_count: 1,
        }
    }

    /// Convert from register to SSA form
    pub fn ssa(&self) -> impl Iterator<Item = SsaOp> + '_ {
        // Map from register index to SSA index
        struct Bindings {
            /// Mapping from register (or memory slot) to SSA index
            data: Vec<u32>,
            /// Next SSA index
            next: u32,
        }
        let mut bound = Bindings {
            data: vec![u32::MAX; self.slot_count as usize],
            next: 0,
        };
        impl Bindings {
            fn reg(&mut self, i: u8) -> u32 {
                if self.data[i as usize] == u32::MAX {
                    self.data[i as usize] = self.next;
                    self.next += 1;
                }
                self.data[i as usize]
            }
            fn out(&mut self, i: u8) -> u32 {
                assert_ne!(self.data[i as usize], u32::MAX);
                let out = self.data[i as usize];
                self.data[i as usize] = u32::MAX;
                out
            }
        }
        bound.reg(0); // bind SSA slot 0 to register 0

        self.tape.iter().filter_map(move |op| {
            let out = match *op {
                RegOp::Load(reg, mem) => {
                    let prev = bound.data[reg as usize];
                    assert_eq!(bound.data[mem as usize], u32::MAX);
                    bound.data[mem as usize] = prev;
                    return None;
                }
                RegOp::Store(reg, mem) => {
                    let prev = bound.data[mem as usize];
                    assert_eq!(bound.data[reg as usize], u32::MAX);
                    bound.data[reg as usize] = prev;
                    return None;
                }
                RegOp::Input(out, i) => SsaOp::Input(bound.out(out), i as u32),
                RegOp::Var(out, i) => SsaOp::Var(bound.out(out), i),
                RegOp::NegReg(out, arg) => {
                    SsaOp::NegReg(bound.out(out), bound.reg(arg))
                }
                RegOp::AbsReg(out, arg) => {
                    SsaOp::AbsReg(bound.out(out), bound.reg(arg))
                }
                RegOp::RecipReg(out, arg) => {
                    SsaOp::RecipReg(bound.out(out), bound.reg(arg))
                }
                RegOp::SqrtReg(out, arg) => {
                    SsaOp::SqrtReg(bound.out(out), bound.reg(arg))
                }
                RegOp::CopyReg(out, arg) => {
                    SsaOp::CopyReg(bound.out(out), bound.reg(arg))
                }
                RegOp::SquareReg(out, arg) => {
                    SsaOp::SquareReg(bound.out(out), bound.reg(arg))
                }
                RegOp::AddRegReg(out, lhs, rhs) => SsaOp::AddRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::MulRegReg(out, lhs, rhs) => SsaOp::MulRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::DivRegReg(out, lhs, rhs) => SsaOp::DivRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::SubRegReg(out, lhs, rhs) => SsaOp::SubRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::MinRegReg(out, lhs, rhs) => SsaOp::MinRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::MaxRegReg(out, lhs, rhs) => SsaOp::MaxRegReg(
                    bound.out(out),
                    bound.reg(lhs),
                    bound.reg(rhs),
                ),
                RegOp::AddRegImm(out, arg, imm) => {
                    SsaOp::AddRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::MulRegImm(out, arg, imm) => {
                    SsaOp::MulRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::DivRegImm(out, arg, imm) => {
                    SsaOp::DivRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::DivImmReg(out, arg, imm) => {
                    SsaOp::DivImmReg(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::SubImmReg(out, arg, imm) => {
                    SsaOp::SubImmReg(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::SubRegImm(out, arg, imm) => {
                    SsaOp::SubRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::MinRegImm(out, arg, imm) => {
                    SsaOp::MinRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::MaxRegImm(out, arg, imm) => {
                    SsaOp::MaxRegImm(bound.out(out), bound.reg(arg), imm)
                }
                RegOp::CopyImm(out, imm) => SsaOp::CopyImm(bound.out(out), imm),
            };
            Some(out)
        })
    }

    /// Resets this tape, retaining its allocations
    pub fn reset(&mut self) {
        self.tape.clear();
        self.slot_count = 1;
    }
    /// Returns the number of unique register and memory locations that are used
    /// by this tape.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count as usize
    }
    /// Returns the number of elements in the tape
    #[inline]
    pub fn len(&self) -> usize {
        self.tape.len()
    }
    /// Returns `true` if the tape contains no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }
    /// Returns a front-to-back iterator
    ///
    /// This is the opposite of evaluation order; it will visit the root of the
    /// tree first, and end at the leaves.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, RegOp> {
        self.into_iter()
    }
    #[inline]
    pub(crate) fn push(&mut self, op: RegOp) {
        self.tape.push(op)
    }
}

impl<'a> IntoIterator for &'a RegTape {
    type Item = &'a RegOp;
    type IntoIter = std::slice::Iter<'a, RegOp>;
    fn into_iter(self) -> Self::IntoIter {
        self.tape.iter()
    }
}
