use thiserror::Error;

/// Universal error type for `jitfive`
#[derive(Error, Debug)]
pub enum Error {
    #[error("node is not present in this `Context`")]
    NoSuchNode,
    #[error("`Context` is empty")]
    EmptyContext,
    #[error("`IndexMap` is empty")]
    EmptyMap,
    #[error("unknown opcode {0}")]
    UnknownOpcode(String),
    #[error("empty file")]
    EmptyFile,
    #[error("The variable was not found in the tree")]
    NoSuchVar,
    #[error("The constant was not found in the tree")]
    NoSuchConst,
    #[error("Error occurred during i/o operation: {0}")]
    Io(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
