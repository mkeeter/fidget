//! Module containing the Fidget universal error type
use crate::var::Var;
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

    /// Variable is missing in the evaluation map
    #[error("variable {0} is missing in the evaluation map")]
    MissingVar(Var),

    /// The given node does not have an associated variable
    #[error("node does not have an associated variable")]
    NotAVar,

    /// The given node is not a constant
    #[error("node is not a constant")]
    NotAConst,

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

    /// Could not solve for matrix pseudo-inverse
    #[error("could not solve for matrix pseudo-inverse: {0}")]
    SingularMatrix(&'static str),

    /// IO error; see inner code for details
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// Each tile must be divisible by subsequent tiles
    #[error("bad tile sizes; {0} is not divisible by {1}")]
    BadTileSize(usize, usize),

    /// Tile size list must be in descending order
    #[error("bad tile order; {0} is not larger than {1}")]
    BadTileOrder(usize, usize),

    /// Tile size list must not be empty
    #[error("tile size list must not be empty")]
    EmptyTileSizes,

    /// Image does not have valid corner values for interpolation
    #[error("image does not have valid corner values for interpolation")]
    BadInterpolation,

    /// Rhai error; see inner code for details
    #[cfg(feature = "rhai")]
    #[error("Rhai parse error")]
    RhaiParseError(#[from] rhai::ParseError),

    /// Rhai error; see inner code for details
    #[cfg(feature = "rhai")]
    #[error("Rhai evaluation error")]
    RhaiEvalError(#[from] rhai::EvalAltResult),

    #[cfg(feature = "jit")]
    /// Dynasm error; see inner code for details
    #[error("dynasm error")]
    DynasmError(#[from] dynasmrt::DynasmError),
}

#[cfg(feature = "rhai")]
impl From<Box<rhai::EvalAltResult>> for Error {
    fn from(e: Box<rhai::EvalAltResult>) -> Self {
        Error::RhaiEvalError(*e)
    }
}
