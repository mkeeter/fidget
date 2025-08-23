//! Tape bytecode format
//!
//! Fidget's bytecode is a packed representation of a [RegTape].  It may be
//! used as the evaluation tape for non-Rust VMs (e.g. an interpreter running on
//! a GPU).  The format is **not stable**; it may change without notice.  Users
//! are recommended to dynamically generate an interpreter function using
//! [`iter_ops`], which associates opcode integers with their names.
//!
//! The bytecode format is a list of little-endian `u32` words, representing
//! tape operations in forward-evaluation order. Each operation in the tape maps
//! to two words.
//!
//! The first two words are always `0xFFFF_FFFF 0x0000_0000`, and the last two
//! words are always `0xFFFF_FFFF 0xFFFF_FFFF`.  Note that this is equivalent to
//! an operation with opcode `0xFF`, which is used for this and other special
//! purposes.
//!
//! ## Register-only operations
//!
//! Register-only operations (i.e. opcodes without an immediate `f32` or `u32`)
//! are packed into a single `u32` as follows:
//!
//! | Byte | Value                                       |
//! |------|---------------------------------------------|
//! | 0    | opcode                                      |
//! | 1    | output register                             |
//! | 2    | first input register                        |
//! | 3    | second input register                       |
//!
//! Depending on the opcode, the input register bytes may not be used.
//!
//! The second word is always `0xFF000000`
//!
//! ## Operations with an `f32` immediate
//!
//! Operations with an `f32` immediate are packed into two `u32` words.
//! The first word is similar to before:
//!
//! | Byte | Value                                       |
//! |------|---------------------------------------------|
//! | 0    | opcode                                      |
//! | 1    | output register                             |
//! | 2    | first input register                        |
//! | 3    | not used                                    |
//!
//! The second word is the `f32` reinterpreted as a `u32`.
//!
//! ## Operations with an `u32` immediate
//!
//! Operations with a `u32` immediate (e.g.
//! [`Load`](crate::compiler::RegOp::Load)) are also packed into two `u32`
//! words.  The first word is what you'd expect:
//!
//! | Byte | Value                                       |
//! |------|---------------------------------------------|
//! | 0    | opcode                                      |
//! | 1    | input or output register                    |
//! | 2    | not used                                    |
//! | 3    | not used                                    |
//!
//! The second word is the `u32` immediate.
//!
//! ## Opcode values
//!
//! Opcode values are generated automatically from [`BytecodeOp`]
//! values, which are one-to-one with [`RegOp`] variants.

use crate::compiler::{BytecodeOp, RegOp, RegTape};
use zerocopy::IntoBytes;

/// Serialized bytecode for external evaluation
pub struct Bytecode {
    /// Maximum register index used by the tape
    pub reg_count: u8,
    /// Number of additional memory slots used for `Load` / `Store` operations
    pub mem_count: u32,
    /// Raw serialized operations
    pub data: Vec<u32>,
}

#[allow(clippy::len_without_is_empty)]
impl Bytecode {
    /// Returns the length of the bytecode tape (in words)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a view of the byte slice
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

impl RegTape {
    /// Converts to bytecode using the format specified in [`bytecode`](crate::bytecode)
    ///
    /// If `reg_limit` is less than `u8::MAX` (i.e. the tape was planned for a
    /// limited set of registers), the first `u8::MAX - reg_limit` memory slots
    /// are instead mapped to registers in the range `reg_limit + 1..=u8::MAX`;
    /// the [`RegOp::Load`] and [`Store`](RegOp::Store) opcodes are replaced
    /// by [`CopyReg`](RegOp::CopyReg).
    pub fn to_bytecode(&self, reg_limit: usize) -> Bytecode {
        // The initial opcode is `OP_JUMP 0x0000_0000`
        let mut data = vec![u32::MAX, 0u32];
        let mut reg_count = 0u8;
        let slot_to_reg = |slot| slot as usize + reg_limit + 1;
        for op in self.iter().rev() {
            let r = BytecodeOp::from(op);
            let mut word = [r as u8, 0xFF, 0xFF, 0xFF];
            let mut imm = None;
            let mut store_reg = |i, r| {
                reg_count = reg_count.max(r); // update the max reg
                word[i] = r;
            };
            match *op {
                RegOp::Input(reg, slot) | RegOp::Output(reg, slot) => {
                    store_reg(1, reg);
                    imm = Some(slot);
                }

                // Patch Load and Store operators if we can use registers intead
                // XXX could we do this without explicit copies, by remapping?
                RegOp::Load(reg, slot)
                    if slot_to_reg(slot) <= u8::MAX as usize =>
                {
                    store_reg(1, reg);
                    store_reg(2, u8::try_from(slot_to_reg(slot)).unwrap());
                    word[0] = BytecodeOp::CopyReg as u8;
                }
                RegOp::Store(reg, slot)
                    if slot_to_reg(slot) <= u8::MAX as usize =>
                {
                    store_reg(1, u8::try_from(slot_to_reg(slot)).unwrap());
                    store_reg(2, reg);
                    word[0] = BytecodeOp::CopyReg as u8;
                }

                RegOp::Load(reg, slot) | RegOp::Store(reg, slot) => {
                    store_reg(1, reg);
                    imm = Some(slot - reg_limit as u32);
                }

                RegOp::CopyImm(out, imm_f32) => {
                    store_reg(1, out);
                    imm = Some(imm_f32.to_bits());
                }
                RegOp::NegReg(out, reg)
                | RegOp::AbsReg(out, reg)
                | RegOp::RecipReg(out, reg)
                | RegOp::SqrtReg(out, reg)
                | RegOp::SquareReg(out, reg)
                | RegOp::FloorReg(out, reg)
                | RegOp::CeilReg(out, reg)
                | RegOp::RoundReg(out, reg)
                | RegOp::CopyReg(out, reg)
                | RegOp::SinReg(out, reg)
                | RegOp::CosReg(out, reg)
                | RegOp::TanReg(out, reg)
                | RegOp::AsinReg(out, reg)
                | RegOp::AcosReg(out, reg)
                | RegOp::AtanReg(out, reg)
                | RegOp::ExpReg(out, reg)
                | RegOp::LnReg(out, reg)
                | RegOp::NotReg(out, reg) => {
                    store_reg(1, out);
                    store_reg(2, reg);
                }

                RegOp::AddRegImm(out, reg, imm_f32)
                | RegOp::MulRegImm(out, reg, imm_f32)
                | RegOp::DivRegImm(out, reg, imm_f32)
                | RegOp::DivImmReg(out, reg, imm_f32)
                | RegOp::SubImmReg(out, reg, imm_f32)
                | RegOp::SubRegImm(out, reg, imm_f32)
                | RegOp::AtanRegImm(out, reg, imm_f32)
                | RegOp::AtanImmReg(out, reg, imm_f32)
                | RegOp::MinRegImm(out, reg, imm_f32)
                | RegOp::MaxRegImm(out, reg, imm_f32)
                | RegOp::CompareRegImm(out, reg, imm_f32)
                | RegOp::CompareImmReg(out, reg, imm_f32)
                | RegOp::ModRegImm(out, reg, imm_f32)
                | RegOp::ModImmReg(out, reg, imm_f32)
                | RegOp::AndRegImm(out, reg, imm_f32)
                | RegOp::OrRegImm(out, reg, imm_f32) => {
                    store_reg(1, out);
                    store_reg(2, reg);
                    imm = Some(imm_f32.to_bits());
                }

                RegOp::AddRegReg(out, lhs, rhs)
                | RegOp::MulRegReg(out, lhs, rhs)
                | RegOp::DivRegReg(out, lhs, rhs)
                | RegOp::SubRegReg(out, lhs, rhs)
                | RegOp::AtanRegReg(out, lhs, rhs)
                | RegOp::MinRegReg(out, lhs, rhs)
                | RegOp::MaxRegReg(out, lhs, rhs)
                | RegOp::CompareRegReg(out, lhs, rhs)
                | RegOp::ModRegReg(out, lhs, rhs)
                | RegOp::AndRegReg(out, lhs, rhs)
                | RegOp::OrRegReg(out, lhs, rhs) => {
                    store_reg(1, out);
                    store_reg(2, lhs);
                    store_reg(3, rhs);
                }
            }
            data.push(u32::from_le_bytes(word));
            data.push(imm.unwrap_or(0xFF000000));
        }
        // Add the final `OP_JUMP 0xFFFF_FFFF`
        data.extend([u32::MAX, u32::MAX]);

        let mem_count = self.slot_count().saturating_sub(u8::MAX as usize);
        Bytecode {
            data,
            mem_count: mem_count as u32,
            reg_count,
        }
    }
}

/// Iterates over opcode `(names, value)` tuples, with names in `CamelCase`
///
/// This is a helper function for defining constants in a VM interpreter
pub fn iter_ops<'a>() -> impl Iterator<Item = (&'a str, u8)> {
    use strum::IntoEnumIterator;

    BytecodeOp::iter().enumerate().map(|(i, op)| {
        let s: &'static str = op.into();
        (s, i as u8)
    })
}
