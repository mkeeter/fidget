//! Canonical bytecode format
//!
//! Fidget's bytecode is a semi-stable, versioned representation of an
//! [RegTape].  It may be used as the evaluation tape for non-Rust VMs (e.g. an
//! interpreter running on a GPU).  The encoding **will not** change without the
//! version being bumped, but is not guaranteed to remain stable between crate
//! releases.  Backwards compatability is not a priority; this encoding is not
//! recommended for on-disk archival purposes.
//!
//! The bytecode format is a list of little-endian `u32` words, representing
//! tape operations in forward-evaluation order. Each tape operation maps to
//! either one or two words.
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
//! Opcode values are generated automatically from [`RegOpDiscriminants`]
//! values.

use crate::compiler::{RegOp, RegOpDiscriminants, RegTape};
use zerocopy::IntoBytes;

/// Current bytecode version
///
/// This will be bumped if the encoding changes, so users of the crate are
/// recommended to assert that it matches.
pub const VERSION: u32 = 1;

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
    pub fn to_bytecode(&self) -> Bytecode {
        let mut data = vec![];

        let mut reg_count = 0u8;
        for op in self.iter().rev() {
            let r = RegOpDiscriminants::from(op);
            let mut word = [r as u8, 0xFF, 0xFF, 0xFF];
            let mut imm = None;
            let mut store_reg = |i, r| {
                reg_count = reg_count.max(r); // update the max reg
                word[i] = r;
            };
            match *op {
                RegOp::Input(reg, slot)
                | RegOp::Output(reg, slot)
                | RegOp::Load(reg, slot)
                | RegOp::Store(reg, slot) => {
                    store_reg(1, reg);
                    imm = Some(slot);
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
            data.extend(imm);
        }

        let mem_count = (self.slot_count() as u32)
            .checked_sub(u32::from(reg_count + 1))
            .unwrap();
        Bytecode {
            data,
            mem_count,
            reg_count,
        }
    }
}

/// Iterates over opcode `(names, value)` tuples
///
/// This is a helper function for defining constants in shaders, so that the
/// shader's bytecode interpreter can use named constants that match the tape.
pub fn iter_ops<'a>() -> impl Iterator<Item = (&'a str, u8)> {
    use strum::IntoEnumIterator;

    RegOpDiscriminants::iter().enumerate().map(|(i, op)| {
        let s: &'static str = op.into();
        (s, i as u8)
    })
}

#[cfg(test)]
mod test {
    use super::*;

    /// Pedantic test for bytecode encoding
    ///
    /// The combination of this test and the exhaustive `match` in
    /// [`RegOp::bytecode_op`](crate::compiler::RegOp::bytecode_op) should make
    /// it impossible to change opcodes or `spec.json` without failing CI.
    #[test]
    fn test_bytecode_encoding() {
        use strum::EnumCount;

        assert_eq!(VERSION, 1);
        assert_eq!(RegOpDiscriminants::COUNT, 50);

        assert_eq!(RegOpDiscriminants::Output as u8, 0);
        assert_eq!(RegOpDiscriminants::Input as u8, 1);
        assert_eq!(RegOpDiscriminants::CopyReg as u8, 2);
        assert_eq!(RegOpDiscriminants::CopyImm as u8, 3);
        assert_eq!(RegOpDiscriminants::NegReg as u8, 4);
        assert_eq!(RegOpDiscriminants::AbsReg as u8, 5);
        assert_eq!(RegOpDiscriminants::RecipReg as u8, 6);
        assert_eq!(RegOpDiscriminants::SqrtReg as u8, 7);
        assert_eq!(RegOpDiscriminants::SquareReg as u8, 8);
        assert_eq!(RegOpDiscriminants::FloorReg as u8, 9);
        assert_eq!(RegOpDiscriminants::CeilReg as u8, 10);
        assert_eq!(RegOpDiscriminants::RoundReg as u8, 11);
        assert_eq!(RegOpDiscriminants::SinReg as u8, 12);
        assert_eq!(RegOpDiscriminants::CosReg as u8, 13);
        assert_eq!(RegOpDiscriminants::TanReg as u8, 14);
        assert_eq!(RegOpDiscriminants::AsinReg as u8, 15);
        assert_eq!(RegOpDiscriminants::AcosReg as u8, 16);
        assert_eq!(RegOpDiscriminants::AtanReg as u8, 17);
        assert_eq!(RegOpDiscriminants::ExpReg as u8, 18);
        assert_eq!(RegOpDiscriminants::LnReg as u8, 19);
        assert_eq!(RegOpDiscriminants::NotReg as u8, 20);
        assert_eq!(RegOpDiscriminants::AddRegImm as u8, 21);
        assert_eq!(RegOpDiscriminants::MulRegImm as u8, 22);
        assert_eq!(RegOpDiscriminants::DivRegImm as u8, 23);
        assert_eq!(RegOpDiscriminants::SubRegImm as u8, 24);
        assert_eq!(RegOpDiscriminants::ModRegImm as u8, 25);
        assert_eq!(RegOpDiscriminants::AtanRegImm as u8, 26);
        assert_eq!(RegOpDiscriminants::CompareRegImm as u8, 27);
        assert_eq!(RegOpDiscriminants::DivImmReg as u8, 28);
        assert_eq!(RegOpDiscriminants::SubImmReg as u8, 29);
        assert_eq!(RegOpDiscriminants::ModImmReg as u8, 30);
        assert_eq!(RegOpDiscriminants::AtanImmReg as u8, 31);
        assert_eq!(RegOpDiscriminants::CompareImmReg as u8, 32);
        assert_eq!(RegOpDiscriminants::MinRegImm as u8, 33);
        assert_eq!(RegOpDiscriminants::MaxRegImm as u8, 34);
        assert_eq!(RegOpDiscriminants::AndRegImm as u8, 35);
        assert_eq!(RegOpDiscriminants::OrRegImm as u8, 36);
        assert_eq!(RegOpDiscriminants::AddRegReg as u8, 37);
        assert_eq!(RegOpDiscriminants::MulRegReg as u8, 38);
        assert_eq!(RegOpDiscriminants::DivRegReg as u8, 39);
        assert_eq!(RegOpDiscriminants::SubRegReg as u8, 40);
        assert_eq!(RegOpDiscriminants::CompareRegReg as u8, 41);
        assert_eq!(RegOpDiscriminants::AtanRegReg as u8, 42);
        assert_eq!(RegOpDiscriminants::ModRegReg as u8, 43);
        assert_eq!(RegOpDiscriminants::MinRegReg as u8, 44);
        assert_eq!(RegOpDiscriminants::MaxRegReg as u8, 45);
        assert_eq!(RegOpDiscriminants::AndRegReg as u8, 46);
        assert_eq!(RegOpDiscriminants::OrRegReg as u8, 47);
        assert_eq!(RegOpDiscriminants::Load as u8, 48);
        assert_eq!(RegOpDiscriminants::Store as u8, 49);
    }
}
