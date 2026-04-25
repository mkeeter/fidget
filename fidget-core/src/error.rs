//! Module containing the Fidget universal error type
use thiserror::Error;

/// Universal error type for Fidget
#[derive(Error, Debug)]
pub enum Error {
    /// Variable index exceeds max var index for this tape
    #[error("variable index ({0}) exceeds max var index for this tape ({1})")]
    BadVarIndex(usize, usize),

    /// Could not solve for matrix pseudo-inverse
    #[error("could not solve for matrix pseudo-inverse: {0}")]
    SingularMatrix(&'static str),
}
