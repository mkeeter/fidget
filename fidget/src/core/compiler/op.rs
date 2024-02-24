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
            #[doc = "Read one of the inputs (X, Y, Z)"]
            Input($t, $t),

            #[doc = "Reads one of the variables"]
            Var($t, u32),

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

            #[doc = "Copies the given register"]
            CopyReg($t, $t),

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
            #[doc = "Subtract an immediate from a register"]
            SubRegImm($t, $t, f32),
            #[doc = "Compute the minimum of a register and an immediate"]
            MinRegImm($t, $t, f32),
            #[doc = "Compute the maximum of a register and an immediate"]
            MaxRegImm($t, $t, f32),

            #[doc = "Add two registers"]
            AddRegReg($t, $t, $t),
            #[doc = "Multiply two registers"]
            MulRegReg($t, $t, $t),
            #[doc = "Divides two registers"]
            DivRegReg($t, $t, $t),
            #[doc = "Subtract one register from another"]
            SubRegReg($t, $t, $t),
            #[doc = "Take the minimum of two registers"]
            MinRegReg($t, $t, $t),
            #[doc = "Take the maximum of two registers"]
            MaxRegReg($t, $t, $t),

            #[doc = "Copy an immediate to a register"]
            CopyImm($t, f32),

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
    #[derive(Copy, Clone, Debug)]
    pub enum SsaOp<u32> {
        // default variants
    }
);

impl RegOp {
    /// Returns the output pseudo-register
    pub fn output(&self) -> u32 {
        match self {
            RegOp::Input(out, ..)
            | RegOp::Var(out, ..)
            | RegOp::CopyImm(out, ..)
            | RegOp::NegReg(out, ..)
            | RegOp::AbsReg(out, ..)
            | RegOp::RecipReg(out, ..)
            | RegOp::SqrtReg(out, ..)
            | RegOp::SquareReg(out, ..)
            | RegOp::CopyReg(out, ..)
            | RegOp::AddRegImm(out, ..)
            | RegOp::MulRegImm(out, ..)
            | RegOp::DivRegImm(out, ..)
            | RegOp::DivImmReg(out, ..)
            | RegOp::SubImmReg(out, ..)
            | RegOp::SubRegImm(out, ..)
            | RegOp::AddRegReg(out, ..)
            | RegOp::MulRegReg(out, ..)
            | RegOp::DivRegReg(out, ..)
            | RegOp::SubRegReg(out, ..)
            | RegOp::MinRegImm(out, ..)
            | RegOp::MaxRegImm(out, ..)
            | RegOp::MinRegReg(out, ..)
            | RegOp::MaxRegReg(out, ..)
            | RegOp::Load(out, ..) => *out as u32,
            RegOp::Store(_reg, mem) => *mem,
        }
    }

    /// Iterates over children (both registers and memory)
    pub fn iter_children(&self) -> impl Iterator<Item = u32> {
        match self {
            RegOp::Input(..) | RegOp::Var(..) | RegOp::CopyImm(..) => {
                [None, None]
            }
            RegOp::NegReg(_out, arg)
            | RegOp::AbsReg(_out, arg)
            | RegOp::RecipReg(_out, arg)
            | RegOp::SqrtReg(_out, arg)
            | RegOp::SquareReg(_out, arg)
            | RegOp::CopyReg(_out, arg)
            | RegOp::AddRegImm(_out, arg, ..)
            | RegOp::MulRegImm(_out, arg, ..)
            | RegOp::DivRegImm(_out, arg, ..)
            | RegOp::DivImmReg(_out, arg, ..)
            | RegOp::SubImmReg(_out, arg, ..)
            | RegOp::SubRegImm(_out, arg, ..)
            | RegOp::MinRegImm(_out, arg, ..)
            | RegOp::MaxRegImm(_out, arg, ..) => [Some(*arg as u32), None],
            RegOp::AddRegReg(_out, lhs, rhs)
            | RegOp::MulRegReg(_out, lhs, rhs)
            | RegOp::DivRegReg(_out, lhs, rhs)
            | RegOp::SubRegReg(_out, lhs, rhs)
            | RegOp::MinRegReg(_out, lhs, rhs)
            | RegOp::MaxRegReg(_out, lhs, rhs) => {
                [Some(*lhs as u32), Some(*rhs as u32)]
            }
            RegOp::Load(_reg, mem) => [Some(*mem), None],
            RegOp::Store(reg, _mem) => [Some(*reg as u32), None],
        }
        .into_iter()
        .flatten()
    }

    /// Returns true if the given opcode is associated with a choice
    pub fn has_choice(&self) -> bool {
        match self {
            RegOp::Input(..)
            | RegOp::Var(..)
            | RegOp::CopyImm(..)
            | RegOp::NegReg(..)
            | RegOp::AbsReg(..)
            | RegOp::RecipReg(..)
            | RegOp::SqrtReg(..)
            | RegOp::SquareReg(..)
            | RegOp::CopyReg(..)
            | RegOp::AddRegImm(..)
            | RegOp::MulRegImm(..)
            | RegOp::SubRegImm(..)
            | RegOp::SubImmReg(..)
            | RegOp::AddRegReg(..)
            | RegOp::MulRegReg(..)
            | RegOp::SubRegReg(..)
            | RegOp::DivRegReg(..)
            | RegOp::DivRegImm(..)
            | RegOp::DivImmReg(..)
            | RegOp::Load(..)
            | RegOp::Store(..) => false,
            RegOp::MinRegImm(..)
            | RegOp::MaxRegImm(..)
            | RegOp::MinRegReg(..)
            | RegOp::MaxRegReg(..) => true,
        }
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
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum RegOp<u8> {
        // default variants
        /// Read from a memory slot to a register
        Load(u8, u32),

        /// Write from a register to a memory slot
        Store(u8, u32),
    }
);
