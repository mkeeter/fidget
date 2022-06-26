use thiserror::Error;

/// Universal error type for `jitfive`
#[derive(Error, Debug)]
pub enum Error {
    #[error("node is not present in this `Context`")]
    BadNode,
    #[error("variable is not present in this `Context`")]
    BadVar,
    #[error("constant is not present in this `Context`")]
    BadConst,

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

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
