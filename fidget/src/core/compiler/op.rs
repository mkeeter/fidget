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

impl SsaOp {
    /// Returns the output pseudo-register
    pub fn output(&self) -> u32 {
        match self {
            SsaOp::Input(out, ..)
            | SsaOp::Var(out, ..)
            | SsaOp::CopyImm(out, ..)
            | SsaOp::NegReg(out, ..)
            | SsaOp::AbsReg(out, ..)
            | SsaOp::RecipReg(out, ..)
            | SsaOp::SqrtReg(out, ..)
            | SsaOp::SquareReg(out, ..)
            | SsaOp::CopyReg(out, ..)
            | SsaOp::AddRegImm(out, ..)
            | SsaOp::MulRegImm(out, ..)
            | SsaOp::DivRegImm(out, ..)
            | SsaOp::DivImmReg(out, ..)
            | SsaOp::SubImmReg(out, ..)
            | SsaOp::SubRegImm(out, ..)
            | SsaOp::AddRegReg(out, ..)
            | SsaOp::MulRegReg(out, ..)
            | SsaOp::DivRegReg(out, ..)
            | SsaOp::SubRegReg(out, ..)
            | SsaOp::MinRegImm(out, ..)
            | SsaOp::MaxRegImm(out, ..)
            | SsaOp::MinRegReg(out, ..)
            | SsaOp::MaxRegReg(out, ..) => *out,
        }
    }
    /// Returns true if the given opcode is associated with a choice
    pub fn has_choice(&self) -> bool {
        match self {
            SsaOp::Input(..)
            | SsaOp::Var(..)
            | SsaOp::CopyImm(..)
            | SsaOp::NegReg(..)
            | SsaOp::AbsReg(..)
            | SsaOp::RecipReg(..)
            | SsaOp::SqrtReg(..)
            | SsaOp::SquareReg(..)
            | SsaOp::CopyReg(..)
            | SsaOp::AddRegImm(..)
            | SsaOp::MulRegImm(..)
            | SsaOp::SubRegImm(..)
            | SsaOp::SubImmReg(..)
            | SsaOp::AddRegReg(..)
            | SsaOp::MulRegReg(..)
            | SsaOp::SubRegReg(..)
            | SsaOp::DivRegReg(..)
            | SsaOp::DivRegImm(..)
            | SsaOp::DivImmReg(..) => false,
            SsaOp::MinRegImm(..)
            | SsaOp::MaxRegImm(..)
            | SsaOp::MinRegReg(..)
            | SsaOp::MaxRegReg(..) => true,
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
