//! Module containing the Fidget universal error type
use thiserror::Error;

/// Universal error type for Fidget
#[derive(Error, Debug)]
pub enum Error {
    #[error("node is not present in this `Context`")]
    BadNode,
    #[error("variable is not present in this `Context`")]
    BadVar,

    #[error("`Context` is empty")]
    EmptyContext,
    #[error("`IndexMap` is empty")]
    EmptyMap,

    #[error("unknown opcode {0}")]
    UnknownOpcode(String),
    #[error("unknown variable {0}")]
    UnknownVariable(String),

    #[error("empty file")]
    EmptyFile,

    #[error("choice slice length ({0}) does not match choice count ({1})")]
    BadChoiceSlice(usize, usize),

    #[error("slice lengths are mismatched")]
    MismatchedSlices,

    #[error("var slice length ({0}) does not match var count ({1})")]
    BadVarSlice(usize, usize),

    #[error("this name is reserved for 3D coordinates")]
    ReservedName,

    #[error("this name has already been used")]
    DuplicateName,

    #[cfg(feature = "rhai")]
    #[error("Rhai error: {0}")]
    RhaiError(#[from] rhai::EvalAltResult),

    #[cfg(feature = "jit")]
    #[error("dynasm error: {0}")]
    DynasmError(#[from] dynasmrt::DynasmError),
}
