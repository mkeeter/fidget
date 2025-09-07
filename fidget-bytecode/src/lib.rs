//! Tape bytecode format
//!
//! Fidget's bytecode is a packed representation of a
//! [`RegTape`](fidget_core::compiler::RegTape).  It may be used as the
//! evaluation tape for non-Rust VMs, e.g. an interpreter running on a GPU.
//!
//! The format is **not stable**; it may change without notice.  It would be
//! wise to dynamically check any interpreter against [`iter_ops`], which
//! associates opcode integers with their names.
//!
//! The bytecode format is a list of little-endian `u32` words, representing
//! tape operations in forward-evaluation order. Each operation in the tape maps
//! to two words, though the second word is not always used.  Having a
//! fixed-length representation makes it easier to iterate both forwards (for
//! evaluation) and backwards (for simplification).
//!
//! The first two words are always `0xFFFF_FFFF 0x0000_0000`, and the last two
//! words are always `0xFFFF_FFFF 0xFFFF_FFFF`.  Note that this is equivalent to
//! an operation with opcode `0xFF`; this special opcode may also be used with
//! user-defined semantics, as long as the immediate is not either reserved
//! value.
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
//! [`Load`](RegOp::Load)) are also packed into two `u32`
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
#![warn(missing_docs)]

use fidget_core::{compiler::RegOp, vm::VmData};
use zerocopy::IntoBytes;

pub use fidget_core::compiler::RegOpDiscriminants as BytecodeOp;

/// Serialized bytecode for external evaluation
pub struct Bytecode {
    reg_count: u8,
    mem_count: u32,
    data: Vec<u32>,
}

impl Bytecode {
    /// Returns the length of the bytecode data (in `u32` words)
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Raw serialized operations
    pub fn data(&self) -> &[u32] {
        &self.data
    }

    /// Maximum register index used by the tape
    pub fn reg_count(&self) -> u8 {
        self.reg_count
    }

    /// Maximum memory slot used for `Load` / `Store` operations
    pub fn mem_count(&self) -> u32 {
        self.mem_count
    }

    /// Returns a view of the byte slice
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Builds a new bytecode object from VM data
    pub fn new<const N: usize>(t: &VmData<N>) -> Self {
        // The initial opcode is `OP_JUMP 0x0000_0000`
        let mut data = vec![u32::MAX, 0u32];
        let mut reg_count = 0u8;
        let mut mem_count = 0u32;
        for op in t.iter_asm() {
            let r = BytecodeOp::from(op);
            let mut word = [r as u8, 0xFF, 0xFF, 0xFF];
            let mut imm = None;
            let mut store_reg = |i, r| {
                reg_count = reg_count.max(r); // update the max reg
                word[i] = r;
            };
            match op {
                RegOp::Input(reg, slot) | RegOp::Output(reg, slot) => {
                    store_reg(1, reg);
                    imm = Some(slot);
                }

                RegOp::Load(reg, slot) | RegOp::Store(reg, slot) => {
                    store_reg(1, reg);
                    mem_count = mem_count.max(slot);
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
            data.push(imm.unwrap_or(0xFF000000));
        }
        // Add the final `OP_JUMP 0xFFFF_FFFF`
        data.extend([u32::MAX, u32::MAX]);

        Bytecode {
            data,
            mem_count,
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple_bytecode() {
        let mut ctx = fidget_core::Context::new();
        let x = ctx.x();
        let c = ctx.constant(1.0);
        let out = ctx.add(x, c).unwrap();
        let data = VmData::<255>::new(&ctx, &[out]).unwrap();
        let bc = Bytecode::new(&data);
        let mut iter = bc.data.iter();
        let mut next = || *iter.next().unwrap();
        assert_eq!(next(), 0xFFFFFFFF); // start marker
        assert_eq!(next(), 0);
        assert_eq!(
            next().to_le_bytes(),
            [BytecodeOp::Input as u8, 0, 0xFF, 0xFF]
        );
        assert_eq!(next(), 0); // input slot 0
        assert_eq!(
            next().to_le_bytes(),
            [BytecodeOp::AddRegImm as u8, 0, 0, 0xFF]
        );
        assert_eq!(f32::from_bits(next()), 1.0);
        assert_eq!(
            next().to_le_bytes(),
            [BytecodeOp::Output as u8, 0, 0xFF, 0xFF]
        );
        assert_eq!(next(), 0); // output slot 0
        assert_eq!(next(), 0xFFFFFFFF); // end marker
        assert_eq!(next(), 0xFFFFFFFF);
        assert!(iter.next().is_none());
    }
}
