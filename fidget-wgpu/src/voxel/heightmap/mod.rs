//! Heightmap evaluation
use crate::{ConfigWorkspace, Gpu, VoxelSizeMismatch, shaders};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const HEIGHTMAP_SHADER: &str = include_str!("shaders/heightmap.wgsl");

fn heightmap_shader() -> String {
    HEIGHTMAP_SHADER.to_owned() + super::PACKED_VOXEL + shaders::COMMON
}

/// GPU-side configuration object for heightmap evaluation
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct HeightmapConfig {
    /// Image size, in pixels
    image_size: [u32; 3],

    /// Flag to indicate whether the color is valid
    has_color: u32,
}

/// Handle to workspace buffers used when rendering a heightmap
pub type Workspace = ConfigWorkspace<HeightmapConfig>;

/// Error returned when submitting a heightmap operation
#[derive(Debug, thiserror::Error)]
pub enum HeightmapError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] VoxelSizeMismatch),
}

/// WGPU context for evaluating a heightmap
pub struct Context {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
}

impl Context {
    /// Builds a new context for applying effects
    pub fn new(gpu: &Gpu) -> Self {
        let shader_code = heightmap_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects heightmap pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout),
                    Some(&gpu.output_bind_group_layout),
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects heightmap shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects heightmap compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("heightmap_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Self {
            gpu: gpu.clone(),
            pipeline,
        }
    }

    /// Builds a new heightmap [`Workspace`]
    pub fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("heightmap")
    }

    /// Submits an operation to generate RGB color from a heightmap
    ///
    /// The output buffer's size must match the input buffer
    pub fn submit(
        &self,
        image: &super::merge::Buffers,
        buf: &mut super::shade::Buffers,
        workspace: &mut Workspace,
    ) -> Result<(), HeightmapError> {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("heightmap compute encoder"),
            },
        );
        let mut staging =
            wgpu::util::StagingBelt::new(self.gpu.device.clone(), 1024);
        self.encode(image, buf, workspace, &mut encoder, &mut staging)?;
        staging.finish();
        self.gpu.queue.submit(Some(encoder.finish()));
        staging.recall();
        Ok(())
    }

    /// Low-level function to encode heightmap rendering
    pub fn encode(
        &self,
        image: &super::merge::Buffers,
        buf: &mut super::shade::Buffers,
        workspace: &mut Workspace,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), HeightmapError> {
        let size = image.out.size();
        if size != buf.out.size() {
            return Err(HeightmapError::ImageSizeMismatch(VoxelSizeMismatch {
                expected: size,
                actual: buf.output().size(),
            }));
        }

        let cfg = HeightmapConfig {
            image_size: [size.width(), size.height(), size.depth()],
            has_color: buf.has_color.into(),
        };
        workspace.encode_copy(encoder, staging, &cfg);

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("heightmap compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, Some(&workspace.bind_group), &[]);
        compute_pass.set_bind_group(1, Some(&image.ro_bind_group), &[]);
        compute_pass.set_bind_group(2, Some(&buf.bind_group), &[]);
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.has_color = false;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_heightmap_shader() {
        crate::compile_shader(&heightmap_shader(), "heightmap");
    }

    #[test]
    fn heightmap_config_layout() {
        crate::test::compare_struct_layout::<HeightmapConfig>(
            &heightmap_shader(),
            "HeightmapConfig",
        );
    }
}
