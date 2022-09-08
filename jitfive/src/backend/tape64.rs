#[derive(Copy, Clone, Debug)]
pub enum ClauseOp64 {
    /// Reads one of the inputs (X, Y, Z)
    Input,

    NegReg,
    AbsReg,
    RecipReg,
    SqrtReg,
    SquareReg,

    /// Copies the given register
    ///
    /// (this is only useful in an `AllocatedTape`)
    CopyReg,

    /// Add a register and an immediate
    AddRegImm,
    /// Multiply a register and an immediate
    MulRegImm,
    /// Subtract a register from an immediate
    SubImmReg,
    /// Subtract an immediate from a register
    SubRegImm,
    /// Compute the minimum of a register and an immediate
    MinRegImm,
    /// Compute the maximum of a register and an immediate
    MaxRegImm,

    AddRegReg,
    MulRegReg,
    SubRegReg,
    MinRegReg,
    MaxRegReg,

    /// Copy an immediate to a register
    CopyImm,
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
pub struct Tape {
    /// The tape is stored in reverse order, such that the root of the tree is
    /// the first item in the tape.
    pub tape: Vec<ClauseOp64>,

    /// Variable-length data for tape clauses.
    ///
    /// Data is densely packed in the order
    /// - output slot
    /// - lhs slot (or input)
    /// - rhs slot (or immediate)
    ///
    /// i.e. a unary operation would only store two items in this array
    pub data: Vec<u32>,

    /// Number of slots used during evaluation
    ///
    /// Slots may not be densely packed!
    pub slot_count: usize,

    /// Number of choice operations in the tape
    pub choice_count: usize,
}
