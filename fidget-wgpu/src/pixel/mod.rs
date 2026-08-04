use crate::{opcode_constants, shaders};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const INTERVAL_INPUT: &str = include_str!("shaders/interval_input.wgsl");
const TRANSFORM_INPUT: &str = include_str!("shaders/transform_input.wgsl");
const INTERVAL_ROOT_SHADER: &str = include_str!("shaders/interval_root.wgsl");
const INTERVAL_TILES_SHADER: &str = include_str!("shaders/interval_tiles.wgsl");
const VOXEL_TILES_SHADER: &str = include_str!("shaders/voxel_tiles.wgsl");

/// Returns a shader for interval root tiles
fn interval_root_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += COMMON_SHADER;
    shader_code += INTERVAL_ROOT_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += TRANSFORM_INPUT;
    shader_code += shaders::INTERVAL_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::STACK;
    shader_code += shaders::TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for interval tile -> subtile reduction
fn interval_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += COMMON_SHADER;
    shader_code += INTERVAL_TILES_SHADER;
    shader_code += INTERVAL_INPUT;
    shader_code += TRANSFORM_INPUT;
    shader_code += shaders::INTERVAL_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::STACK;
    shader_code += shaders::TAPE_SIMPLIFY;
    shader_code
}

/// Returns a shader for voxel tile evaluation
fn voxel_tiles_shader(reg_count: u8) -> String {
    let mut shader_code = opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code += VOXEL_TILES_SHADER;
    shader_code += TRANSFORM_INPUT;
    shader_code += COMMON_SHADER;
    shader_code += shaders::FLOAT_OPS;
    shader_code += shaders::COMMON;
    shader_code += shaders::TAPE_INTERPRETER;
    shader_code += shaders::DUMMY_STACK;
    shader_code
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_shaders() {
        for (src, desc) in [
            (interval_root_shader(16), "interval root"),
            (interval_tiles_shader(16), "interval tiles"),
            (voxel_tiles_shader(16), "voxel tiles"),
        ] {
            crate::compile_shader(&src, desc);
        }
    }
}
