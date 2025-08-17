//! Shader generation and WGPU-based image rendering
pub mod render3d;
pub(crate) mod util;

use crate::bytecode;
use heck::ToShoutySnakeCase;

/// Returns a set of constant definitions for each opcode
fn opcode_constants() -> String {
    let mut out = String::new();
    for (op, i) in bytecode::iter_ops() {
        out += &format!("const OP_{}: u32 = {i};\n", op.to_shouty_snake_case());
    }
    out
}
