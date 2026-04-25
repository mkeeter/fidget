//! Module containing the Fidget universal error type
use thiserror::Error;

/// Universal error type for Fidget
#[derive(Error, Debug)]
pub enum Error {
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
}
