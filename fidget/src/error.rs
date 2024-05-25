//! Module containing the Fidget universal error type
use thiserror::Error;

/// Universal error type for Fidget
#[derive(Error, Debug)]
pub enum Error {
    /// Node is not present in this `Context`
    #[error("node is not present in this `Context`")]
    BadNode,
    /// Variable is not present in this `Context`
    #[error("variable is not present in this `Context`")]
    BadVar,

    /// The given node does not have an associated variable
    #[error("node does not have an associated variable")]
    NotAVar,

    /// `Context` is empty
    #[error("`Context` is empty")]
    EmptyContext,
    /// `IndexMap` is empty
    #[error("`IndexMap` is empty")]
    EmptyMap,

    /// Unknown opcode {0}
    #[error("unknown opcode {0}")]
    UnknownOpcode(String),
    /// Unknown variable {0}
    #[error("unknown variable {0}")]
    UnknownVariable(String),

    /// Empty file
    #[error("empty file")]
    EmptyFile,

    /// Choice slice length does not match choice count
    #[error("choice slice length ({0}) does not match choice count ({1})")]
    BadChoiceSlice(usize, usize),

    /// Variable slice lengths are mismatched
    #[error("variable slice lengths are mismatched")]
    MismatchedSlices,

    /// Variable slice length does not match expected count
    #[error("variable slice length ({0}) does not match expected count ({1})")]
    BadVarSlice(usize, usize),

    /// Variable index exceeds max var index for this tape
    #[error("variable index ({0}) exceeds max var index for this tape ({1})")]
    BadVarIndex(usize, usize),

    /// This name is reserved for 3D coordinates
    #[error("this name is reserved for 3D coordinates")]
    ReservedName,

    /// This name has already been used
    #[error("this name has already been used")]
    DuplicateName,

    /// io error; see inner code for details
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[cfg(feature = "rhai")]
    /// Rhai error; see inner code for details
    #[error("Rhai evaluation error: {0}")]
    RhaiParseError(#[from] rhai::ParseError),

    #[cfg(feature = "rhai")]
    /// Rhai error; see inner code for details
    #[error("Rhai evaluation error: {0}")]
    RhaiEvalError(#[from] rhai::EvalAltResult),

    #[cfg(feature = "jit")]
    /// Dynasm error; see inner code for details
    #[error("dynasm error: {0}")]
    DynasmError(#[from] dynasmrt::DynasmError),
}

#[cfg(feature = "rhai")]
impl From<Box<rhai::EvalAltResult>> for Error {
    fn from(e: Box<rhai::EvalAltResult>) -> Self {
        Error::RhaiEvalError(*e)
    }
}
