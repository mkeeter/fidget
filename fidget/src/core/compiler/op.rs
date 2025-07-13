use serde::{Deserialize, Serialize};

/// Macro to generate a set of opcodes, using the given type for registers
macro_rules! opcodes {
    (
        $(#[$($attrss:meta)*])*
        pub enum $name:ident<$t:ty> {}
    ) => {
        opcodes!(
            $(#[$($attrss)*])*
            pub enum $name<$t> {,}
        );
    };
    (
        $(#[$($attrss:meta)*])*
        pub enum $name:ident<$t:ty> {
            $(
                $(#[$($a:meta)*])*
                $foo:ident($($i:ty),*)
             ),*
            ,
        }
    ) => {
        $(#[$($attrss)*])*
        pub enum $name {
            // Special unary opcodes
            #[doc = "Writes an output variable by index"]
            Output($t, u32),
            #[doc = "Read an input variable by index"]
            Input($t, u32),
            #[doc = "Copies the given register"]
            CopyReg($t, $t),
            #[doc = "Copy an immediate to a register"]
            CopyImm($t, f32),

            // Normal unary opcodes
            #[doc = "Negate the given register"]
            NegReg($t, $t),
            #[doc = "Take the absolute value of the given register"]
            AbsReg($t, $t),
            #[doc = "Take the reciprocal of the given register (1.0 / value)"]
            RecipReg($t, $t),
            #[doc = "Take the square root of the given register"]
            SqrtReg($t, $t),
            #[doc = "Square the given register"]
            SquareReg($t, $t),
            #[doc = "Returns the largest integer less than or equal to `self`"]
            FloorReg($t, $t),
            #[doc = "Returns the smallest integer greater than or equal to `self`"]
            CeilReg($t, $t),
            #[doc = "Returns the nearest integer to `self`. If a value is half-way between two integers, round away from `0.0`."]
            RoundReg($t, $t),
            #[doc = "Computes the sine of the given register (in radians)"]
            SinReg($t, $t),
            #[doc = "Computes the cosine of the given register (in radians)"]
            CosReg($t, $t),
            #[doc = "Computes the tangent of the given register (in radians)"]
            TanReg($t, $t),
            #[doc = "Computes the arcsin of the given register (in radians)"]
            AsinReg($t, $t),
            #[doc = "Computes the arccos of the given register (in radians)"]
            AcosReg($t, $t),
            #[doc = "Computes the arctangent of the given register (in radians)"]
            AtanReg($t, $t),
            #[doc = "Computes the exponential function of the given register"]
            ExpReg($t, $t),
            #[doc = "Computes the natural log of the given register"]
            LnReg($t, $t),
            #[doc = "Computes the logical negation of the given register\n\nEquivalent to `if arg == 0 { 1 } else { 0 }`"]
            NotReg($t, $t),

            // RegImm opcodes (without a choice)
            #[doc = "Add a register and an immediate"]
            AddRegImm($t, $t, f32),
            #[doc = "Multiply a register and an immediate"]
            MulRegImm($t, $t, f32),
            #[doc = "Divides a register and an immediate"]
            DivRegImm($t, $t, f32),
            #[doc = "Divides an immediate by a register"]
            DivImmReg($t, $t, f32),
            #[doc = "Subtract a register from an immediate"]
            SubImmReg($t, $t, f32),
            #[doc = "Take the module (least nonnegative remainder) of two registers"]
            ModRegReg($t, $t, $t),
            #[doc = "Take the module (least nonnegative remainder) of a register and an immediate"]
            ModRegImm($t, $t, f32),
            #[doc = "atan2 of a position `(y, x)` specified as register, immediate"]
            AtanRegImm($t, $t, f32),
            #[doc = "Compares a register with an immediate"]
            CompareRegImm($t, $t, f32),

            // RegImm opcodes (with a choice)
            #[doc = "Compute the minimum of a register and an immediate"]
            MinRegImm($t, $t, f32),
            #[doc = "Compute the maximum of a register and an immediate"]
            MaxRegImm($t, $t, f32),
            #[doc = "Logical `AND` (short-circuiting)\n\nThis is equivalent to `if lhs == 0 { lhs } else { imm }`"]
            AndRegImm($t, $t, f32),
            #[doc = "Logical `OR` (short-circuiting)\n\nThis is equivalent to `if lhs != 0 { lhs } else { imm }`"]
            OrRegImm($t, $t, f32),

            // ImmReg opcodes (without a choice)
            #[doc = "Take the module (least nonnegative remainder) of an immediate and a register"]
            ModImmReg($t, $t, f32),
            #[doc = "atan2 of a position `(y, x)` specified as immediate, register"]
            AtanImmReg($t, $t, f32),
            #[doc = "Compares an immediate with a register"]
            CompareImmReg($t, $t, f32),

            // RegReg opcodes (without a choice)
            #[doc = "Add two registers"]
            AddRegReg($t, $t, $t),
            #[doc = "Multiply two registers"]
            MulRegReg($t, $t, $t),
            #[doc = "Divides two registers"]
            DivRegReg($t, $t, $t),
            #[doc = "Subtract one register from another"]
            SubRegReg($t, $t, $t),
            #[doc = "Compares two registers"]
            CompareRegReg($t, $t, $t),
            #[doc = "atan2 of a position `(y, x)` specified as register, register"]
            AtanRegReg($t, $t, $t),

            // Higher-level operations (generated by a peephole optimizer)
            #[doc = "radius of a 2D position"]
            RadiusRegReg($t, $t, $t),
            #[doc = "radius of a 2D position, minus a constant"]
            RadiusRegRegImm($t, $t, $t, f32),

            // RegReg opcodes (with a choice)
            #[doc = "Take the minimum of two registers"]
            MinRegReg($t, $t, $t),
            #[doc = "Take the maximum of two registers"]
            MaxRegReg($t, $t, $t),
            #[doc = "Logical `AND` (short-circuiting)\n\nThis is equivalent to `if lhs == 0 { lhs } else { rhs }`"]
            AndRegReg($t, $t, $t),
            #[doc = "Logical `OR` (short-circuiting)\n\nThis is equivalent to `if lhs != 0 { lhs } else { rhs }`"]
            OrRegReg($t, $t, $t),

            $(
                $(#[$($a)*])*
                $foo($($i),*)
             ),*
        }
    };
}

opcodes!(
    /// Basic operations that can be performed in a tape
    ///
    /// Arguments, in order, are
    /// - Output register
    /// - LHS register (or input slot for [`Input`](SsaOp::Input))
    /// - RHS register (or immediate for `*Imm`)
    ///
    /// Each "register" represents an SSA slot, which is never reused.
    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    pub enum SsaOp<u32> {
        // default variants
    }
);

impl SsaOp {
    /// Returns the output pseudo-register
    pub fn output(&self) -> Option<u32> {
        match self {
            SsaOp::Input(out, ..)
            | SsaOp::CopyImm(out, ..)
            | SsaOp::NegReg(out, ..)
            | SsaOp::AbsReg(out, ..)
            | SsaOp::RecipReg(out, ..)
            | SsaOp::SqrtReg(out, ..)
            | SsaOp::SquareReg(out, ..)
            | SsaOp::FloorReg(out, ..)
            | SsaOp::CeilReg(out, ..)
            | SsaOp::RoundReg(out, ..)
            | SsaOp::CopyReg(out, ..)
            | SsaOp::SinReg(out, ..)
            | SsaOp::CosReg(out, ..)
            | SsaOp::TanReg(out, ..)
            | SsaOp::AsinReg(out, ..)
            | SsaOp::AcosReg(out, ..)
            | SsaOp::AtanReg(out, ..)
            | SsaOp::ExpReg(out, ..)
            | SsaOp::LnReg(out, ..)
            | SsaOp::NotReg(out, ..)
            | SsaOp::AddRegImm(out, ..)
            | SsaOp::MulRegImm(out, ..)
            | SsaOp::DivRegImm(out, ..)
            | SsaOp::DivImmReg(out, ..)
            | SsaOp::SubImmReg(out, ..)
            | SsaOp::AddRegReg(out, ..)
            | SsaOp::MulRegReg(out, ..)
            | SsaOp::DivRegReg(out, ..)
            | SsaOp::SubRegReg(out, ..)
            | SsaOp::AtanRegReg(out, ..)
            | SsaOp::AtanRegImm(out, ..)
            | SsaOp::AtanImmReg(out, ..)
            | SsaOp::MinRegImm(out, ..)
            | SsaOp::MaxRegImm(out, ..)
            | SsaOp::MinRegReg(out, ..)
            | SsaOp::MaxRegReg(out, ..)
            | SsaOp::CompareRegReg(out, ..)
            | SsaOp::CompareRegImm(out, ..)
            | SsaOp::CompareImmReg(out, ..)
            | SsaOp::ModRegReg(out, ..)
            | SsaOp::ModRegImm(out, ..)
            | SsaOp::ModImmReg(out, ..)
            | SsaOp::AndRegImm(out, ..)
            | SsaOp::AndRegReg(out, ..)
            | SsaOp::OrRegImm(out, ..)
            | SsaOp::OrRegReg(out, ..)
            | SsaOp::RadiusRegReg(out, ..)
            | SsaOp::RadiusRegRegImm(out, ..) => Some(*out),
            SsaOp::Output(..) => None,
        }
    }
    /// Returns a mutable reference to the output pseudo-register
    pub fn output_mut(&mut self) -> Option<&mut u32> {
        match self {
            SsaOp::Input(out, ..)
            | SsaOp::CopyImm(out, ..)
            | SsaOp::NegReg(out, ..)
            | SsaOp::AbsReg(out, ..)
            | SsaOp::RecipReg(out, ..)
            | SsaOp::SqrtReg(out, ..)
            | SsaOp::SquareReg(out, ..)
            | SsaOp::FloorReg(out, ..)
            | SsaOp::CeilReg(out, ..)
            | SsaOp::RoundReg(out, ..)
            | SsaOp::CopyReg(out, ..)
            | SsaOp::SinReg(out, ..)
            | SsaOp::CosReg(out, ..)
            | SsaOp::TanReg(out, ..)
            | SsaOp::AsinReg(out, ..)
            | SsaOp::AcosReg(out, ..)
            | SsaOp::AtanReg(out, ..)
            | SsaOp::ExpReg(out, ..)
            | SsaOp::LnReg(out, ..)
            | SsaOp::NotReg(out, ..)
            | SsaOp::AddRegImm(out, ..)
            | SsaOp::MulRegImm(out, ..)
            | SsaOp::DivRegImm(out, ..)
            | SsaOp::DivImmReg(out, ..)
            | SsaOp::SubImmReg(out, ..)
            | SsaOp::AddRegReg(out, ..)
            | SsaOp::MulRegReg(out, ..)
            | SsaOp::DivRegReg(out, ..)
            | SsaOp::SubRegReg(out, ..)
            | SsaOp::AtanRegReg(out, ..)
            | SsaOp::AtanRegImm(out, ..)
            | SsaOp::AtanImmReg(out, ..)
            | SsaOp::MinRegImm(out, ..)
            | SsaOp::MaxRegImm(out, ..)
            | SsaOp::MinRegReg(out, ..)
            | SsaOp::MaxRegReg(out, ..)
            | SsaOp::CompareRegReg(out, ..)
            | SsaOp::CompareRegImm(out, ..)
            | SsaOp::CompareImmReg(out, ..)
            | SsaOp::ModRegReg(out, ..)
            | SsaOp::ModRegImm(out, ..)
            | SsaOp::ModImmReg(out, ..)
            | SsaOp::AndRegImm(out, ..)
            | SsaOp::AndRegReg(out, ..)
            | SsaOp::OrRegImm(out, ..)
            | SsaOp::OrRegReg(out, ..)
            | SsaOp::RadiusRegReg(out, ..)
            | SsaOp::RadiusRegRegImm(out, ..) => Some(out),
            SsaOp::Output(..) => None,
        }
    }

    /// Returns true if the given opcode is associated with a choice
    pub fn has_choice(&self) -> bool {
        match self {
            SsaOp::Input(..)
            | SsaOp::Output(..)
            | SsaOp::CopyImm(..)
            | SsaOp::NegReg(..)
            | SsaOp::AbsReg(..)
            | SsaOp::RecipReg(..)
            | SsaOp::SqrtReg(..)
            | SsaOp::SquareReg(..)
            | SsaOp::FloorReg(..)
            | SsaOp::CeilReg(..)
            | SsaOp::RoundReg(..)
            | SsaOp::CopyReg(..)
            | SsaOp::SinReg(..)
            | SsaOp::CosReg(..)
            | SsaOp::TanReg(..)
            | SsaOp::AsinReg(..)
            | SsaOp::AcosReg(..)
            | SsaOp::AtanReg(..)
            | SsaOp::ExpReg(..)
            | SsaOp::LnReg(..)
            | SsaOp::NotReg(..)
            | SsaOp::AddRegImm(..)
            | SsaOp::MulRegImm(..)
            | SsaOp::SubImmReg(..)
            | SsaOp::AddRegReg(..)
            | SsaOp::MulRegReg(..)
            | SsaOp::SubRegReg(..)
            | SsaOp::DivRegReg(..)
            | SsaOp::DivRegImm(..)
            | SsaOp::DivImmReg(..)
            | SsaOp::AtanRegReg(..)
            | SsaOp::AtanRegImm(..)
            | SsaOp::AtanImmReg(..)
            | SsaOp::CompareRegReg(..)
            | SsaOp::CompareRegImm(..)
            | SsaOp::CompareImmReg(..)
            | SsaOp::ModRegReg(..)
            | SsaOp::ModRegImm(..)
            | SsaOp::ModImmReg(..)
            | SsaOp::RadiusRegReg(..)
            | SsaOp::RadiusRegRegImm(..) => false,
            SsaOp::MinRegImm(..)
            | SsaOp::MaxRegImm(..)
            | SsaOp::MinRegReg(..)
            | SsaOp::MaxRegReg(..)
            | SsaOp::AndRegImm(..)
            | SsaOp::AndRegReg(..)
            | SsaOp::OrRegImm(..)
            | SsaOp::OrRegReg(..) => true,
        }
    }

    /// Iterates over arguments
    pub fn iter_children(&self) -> impl Iterator<Item = u32> {
        match self {
            SsaOp::Input(_, _) | SsaOp::CopyImm(_, _) => [None, None],
            SsaOp::NegReg(_, arg)
            | SsaOp::AbsReg(_, arg)
            | SsaOp::RecipReg(_, arg)
            | SsaOp::SqrtReg(_, arg)
            | SsaOp::SquareReg(_, arg)
            | SsaOp::FloorReg(_, arg)
            | SsaOp::CeilReg(_, arg)
            | SsaOp::RoundReg(_, arg)
            | SsaOp::CopyReg(_, arg)
            | SsaOp::SinReg(_, arg)
            | SsaOp::CosReg(_, arg)
            | SsaOp::TanReg(_, arg)
            | SsaOp::AsinReg(_, arg)
            | SsaOp::AcosReg(_, arg)
            | SsaOp::AtanReg(_, arg)
            | SsaOp::ExpReg(_, arg)
            | SsaOp::LnReg(_, arg)
            | SsaOp::NotReg(_, arg)
            | SsaOp::Output(arg, _)
            | SsaOp::MulRegImm(_, arg, _)
            | SsaOp::DivRegImm(_, arg, _)
            | SsaOp::AddRegImm(_, arg, _)
            | SsaOp::DivImmReg(_, arg, _)
            | SsaOp::SubImmReg(_, arg, _)
            | SsaOp::AtanRegImm(_, arg, _)
            | SsaOp::AtanImmReg(_, arg, _)
            | SsaOp::MinRegImm(_, arg, _)
            | SsaOp::MaxRegImm(_, arg, _)
            | SsaOp::CompareRegImm(_, arg, _)
            | SsaOp::CompareImmReg(_, arg, _)
            | SsaOp::ModRegImm(_, arg, _)
            | SsaOp::ModImmReg(_, arg, _)
            | SsaOp::AndRegImm(_, arg, _)
            | SsaOp::OrRegImm(_, arg, _) => [Some(*arg), None],
            SsaOp::AddRegReg(_, lhs, rhs)
            | SsaOp::MulRegReg(_, lhs, rhs)
            | SsaOp::DivRegReg(_, lhs, rhs)
            | SsaOp::SubRegReg(_, lhs, rhs)
            | SsaOp::AtanRegReg(_, lhs, rhs)
            | SsaOp::MinRegReg(_, lhs, rhs)
            | SsaOp::MaxRegReg(_, lhs, rhs)
            | SsaOp::CompareRegReg(_, lhs, rhs)
            | SsaOp::ModRegReg(_, lhs, rhs)
            | SsaOp::AndRegReg(_, lhs, rhs)
            | SsaOp::OrRegReg(_, lhs, rhs)
            | SsaOp::RadiusRegReg(_, lhs, rhs)
            | SsaOp::RadiusRegRegImm(_, lhs, rhs, _) => {
                [Some(*lhs), Some(*rhs)]
            }
        }
        .into_iter()
        .flatten()
    }

    /// Mutable iterator over arguments
    pub fn iter_children_mut(&mut self) -> impl Iterator<Item = &mut u32> {
        match self {
            SsaOp::Input(_, _) | SsaOp::CopyImm(_, _) => [None, None],
            SsaOp::NegReg(_, arg)
            | SsaOp::AbsReg(_, arg)
            | SsaOp::RecipReg(_, arg)
            | SsaOp::SqrtReg(_, arg)
            | SsaOp::SquareReg(_, arg)
            | SsaOp::FloorReg(_, arg)
            | SsaOp::CeilReg(_, arg)
            | SsaOp::RoundReg(_, arg)
            | SsaOp::CopyReg(_, arg)
            | SsaOp::SinReg(_, arg)
            | SsaOp::CosReg(_, arg)
            | SsaOp::TanReg(_, arg)
            | SsaOp::AsinReg(_, arg)
            | SsaOp::AcosReg(_, arg)
            | SsaOp::AtanReg(_, arg)
            | SsaOp::ExpReg(_, arg)
            | SsaOp::LnReg(_, arg)
            | SsaOp::NotReg(_, arg)
            | SsaOp::Output(arg, _)
            | SsaOp::MulRegImm(_, arg, _)
            | SsaOp::DivRegImm(_, arg, _)
            | SsaOp::AddRegImm(_, arg, _)
            | SsaOp::DivImmReg(_, arg, _)
            | SsaOp::SubImmReg(_, arg, _)
            | SsaOp::AtanRegImm(_, arg, _)
            | SsaOp::AtanImmReg(_, arg, _)
            | SsaOp::MinRegImm(_, arg, _)
            | SsaOp::MaxRegImm(_, arg, _)
            | SsaOp::CompareRegImm(_, arg, _)
            | SsaOp::CompareImmReg(_, arg, _)
            | SsaOp::ModRegImm(_, arg, _)
            | SsaOp::ModImmReg(_, arg, _)
            | SsaOp::AndRegImm(_, arg, _)
            | SsaOp::OrRegImm(_, arg, _) => [Some(arg), None],
            SsaOp::AddRegReg(_, lhs, rhs)
            | SsaOp::MulRegReg(_, lhs, rhs)
            | SsaOp::DivRegReg(_, lhs, rhs)
            | SsaOp::SubRegReg(_, lhs, rhs)
            | SsaOp::AtanRegReg(_, lhs, rhs)
            | SsaOp::MinRegReg(_, lhs, rhs)
            | SsaOp::MaxRegReg(_, lhs, rhs)
            | SsaOp::CompareRegReg(_, lhs, rhs)
            | SsaOp::ModRegReg(_, lhs, rhs)
            | SsaOp::AndRegReg(_, lhs, rhs)
            | SsaOp::OrRegReg(_, lhs, rhs)
            | SsaOp::RadiusRegReg(_, lhs, rhs)
            | SsaOp::RadiusRegRegImm(_, lhs, rhs, _) => [Some(lhs), Some(rhs)],
        }
        .into_iter()
        .flatten()
    }
}

opcodes!(
    /// Operations used in register-allocated tapes
    ///
    /// Arguments, in order, are
    /// - Output register
    /// - LHS register (or input slot for [`Input`](RegOp::Input))
    /// - RHS register (or immediate for `*Imm`)
    ///
    /// We have a maximum of 256 registers, though some tapes (e.g. ones
    /// targeting physical hardware) may choose to use fewer.
    #[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub enum RegOp<u8> {
        // default variants
        /// Read from a memory slot to a register
        Load(u8, u32),

        /// Write from a register to a memory slot
        Store(u8, u32),
    }
);
