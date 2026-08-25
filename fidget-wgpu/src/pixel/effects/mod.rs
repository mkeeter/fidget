use crate::{
    Gpu,
    RegPipeline,
    RenderShape,
    TAPE_DATA_CAPACITY,
    TapeWord,
    buf::{
        ArrayBuffer, BufferItemCount, BufferSizeError, BufferType, ImageBuffer,
        buffer_ro, buffer_ro_dyn, buffer_rw,
    },
    shaders,
    tag,
    voxel::effects::MergeConfig, // same layout, unit tested to confirm
};

const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

/// Returns a shader for merging images
fn merge_shader() -> String {
    MERGE_SHADER.to_owned()
        + COMMON_SHADER
        + shaders::COMMON
        + super::DISTANCE_PIXEL_SHADER
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_merge_shader() {
        crate::compile_shader(&merge_shader(), "merge");
    }

    #[test]
    fn merge_config_layout() {
        crate::test::compare_struct_layout::<MergeConfig>(
            &merge_shader(),
            "MergeConfig",
        );
    }
}
