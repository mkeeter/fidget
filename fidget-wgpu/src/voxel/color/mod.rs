//! Color image evaluation
use crate::{
    ConfigWorkspace, Gpu, RegPipeline, VoxelSizeMismatch, color::RenderColor,
    shaders,
};
use fidget_core::shape::{MissingVar, ShapeVars};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const COLOR_SHADER: &str = include_str!("shaders/color.wgsl");

fn color_shader(reg_count: u8) -> String {
    let mut shader_code = shaders::opcode_constants();
    shader_code += &format!("const REG_COUNT: u32 = {reg_count};");
    shader_code
        + COLOR_SHADER
        + super::TRANSFORM_INPUT
        + super::PACKED_VOXEL
        + shaders::COMMON
        + shaders::TAPE_INTERPRETER
        + shaders::DUMMY_STACK
        + shaders::FLOAT_OPS
}

/// Context for rendering color (RGBA) images
pub struct Context {
    gpu: Gpu,
    /// Pipeline for computing per-pixel color
    color_pipeline: RegPipeline,
}

impl Context {
    /// Builds a new color context
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("color pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.color_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
                ],
                immediate_size: 0u32,
            });
        let device_ = device.clone();
        let color_pipeline = RegPipeline::build(Box::new(move |reg_count| {
            let shader_code = color_shader(reg_count);
            let shader_module =
                device_.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
            device_.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("color ({reg_count})")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("color_main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }));

        Self {
            gpu: gpu.clone(),
            color_pipeline,
        }
    }

    /// Submits a color evaluation pass
    ///
    /// Image size is set from the `MergeWorkspace`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit(
        &self,
        merge: &super::merge::Buffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &RenderColor,
        workspace: &mut Workspace,
        out: &mut super::shade::Buffers,
    ) -> Result<(), ColorError> {
        self.submit_with_vars(
            merge,
            world_to_model,
            shape,
            workspace,
            &Default::default(),
            out,
        )
    }

    /// Submits a color evaluation pass with auxiliary variables
    ///
    /// Image size is set from the `MergeWorkspace`; the transform matrix is
    /// provided separately (but should be the same one used for image
    /// evaluation).
    pub fn submit_with_vars(
        &self,
        merge: &super::merge::Buffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &RenderColor,
        workspace: &mut Workspace,
        vars: &ShapeVars<f32>,
        out: &mut super::shade::Buffers,
    ) -> Result<(), ColorError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode(
            merge,
            world_to_model,
            shape,
            workspace,
            vars,
            out,
            &mut encoder,
            &mut staging,
        )?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // what are ya gonna do?
    fn encode(
        &self,
        image: &super::merge::Buffers,
        world_to_model: &nalgebra::Matrix4<f32>,
        shape: &RenderColor,
        workspace: &mut Workspace,
        vars: &ShapeVars<f32>,
        out: &mut super::shade::Buffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ColorError> {
        if image.image_count != shape.shape_count() {
            return Err(ColorError::BadShapeCount {
                merge_count: image.image_count,
                shape_count: shape.shape_count(),
            });
        }
        let size = image.out.size();
        if size != out.out.size() {
            return Err(VoxelSizeMismatch {
                expected: size,
                actual: out.out.size(),
            }
            .into());
        }
        shape.copy_vars(encoder, staging, vars)?;

        out.has_color = true;
        let mat = world_to_model * size.screen_to_world();
        let config = ColorConfig {
            mat: mat.data.as_slice().try_into().unwrap(),
            axes: shape.axes(),
            image_size: [size.width(), size.height()],
            _pad1: 0,
            _pad2: [0; 2],
        };
        workspace.encode_copy(encoder, staging, &config);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, shape.bind_group(), &[]);
        compute_pass.set_bind_group(2, &image.ro_bind_group, &[]);
        compute_pass.set_bind_group(3, &out.bind_group, &[]);

        compute_pass.set_pipeline(self.color_pipeline.get(shape.reg_count()));
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        Ok(())
    }

    /// Returns a new workspace for color evaluation
    pub fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("color")
    }
}

/// Workspace for evaluating voxel color expressions
pub type Workspace = ConfigWorkspace<ColorConfig>;

/// Configuration for the color evaluation pass
///
/// This is quietly public because it's part of a public API, but is unlikely to
/// be useful for end-users of the library.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct ColorConfig {
    /// Screen-to-model transform matrix
    mat: [f32; 16],

    /// Input index of X, Y, Z axes
    ///
    /// `u32::MAX` is used as a marker if an axis is unused
    axes: [u32; 3],

    // padding for alignment
    _pad1: u32,

    /// Image size (pixels)
    image_size: [u32; 2],

    // padding for alignment
    _pad2: [u32; 2],
}

/// Error returned when submitting a color evaluation operation
#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    /// A variable is missing from the map
    #[error(transparent)]
    MissingVar(#[from] MissingVar),

    /// Wrong number of shapes
    #[error(
        "merged image is built from {merge_count} shapes,
         but shape color buffer expected {shape_count} shapes"
    )]
    BadShapeCount {
        /// Number of shapes in the merged image
        merge_count: usize,
        /// Number of shapes in the color buffer
        shape_count: usize,
    },

    /// The input and output buffers have mismatched sizes
    #[error(transparent)]
    SizeMismatch(#[from] VoxelSizeMismatch),

    /// Shape color has already been populated
    #[error("shape color has already been populated")]
    AlreadyHasColor,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn color_config_layout() {
        crate::test::compare_struct_layout::<ColorConfig>(
            &color_shader(16),
            "Config",
        );
    }

    #[test]
    fn compile_color_shader() {
        crate::compile_shader(&color_shader(16), "color");
    }
}
