//! Shader generation and WGPU-based image rendering
pub mod render3d;
pub(crate) mod util;

use heck::ToShoutySnakeCase;

/// Returns a set of constant definitions for each opcode
fn opcode_constants() -> String {
    let mut out = String::new();
    for (op, i) in fidget_bytecode::iter_ops() {
        out += &format!("const OP_{}: u32 = {i};\n", op.to_shouty_snake_case());
    }
    out
}

/// Error type for type construction
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Could not get WGPU adapter
    #[error("could not get adapter")]
    NoAdapter,

    /// Could not get WGPU device
    #[error("could not get WGPU device")]
    NoDevice(#[from] wgpu::RequestDeviceError),
}
