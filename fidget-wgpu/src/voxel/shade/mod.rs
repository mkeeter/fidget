//! Shading post-processing pass
//!
//! Applies shading to a [`PackedVoxel`](crate::voxel::PackedVoxel) buffer,
//! producing an RGBA image buffer
use crate::{
    ConfigWorkspace, Gpu, ImageSizeMismatch, VoxelSizeMismatch,
    buf::{BufferSizeError, FlexBuffer},
    shaders, tag,
};
use fidget_core::render::VoxelSize;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// GPU-side configuration object for shading
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct ShadeConfig {
    /// Image size, in pixels
    image_size: [u32; 3],

    /// Flags to set the presence of color and SSAO
    flags: u32,
}

/// Must match constants in `shade.wgsl`
const SHADE_CONFIG_HAS_SSAO: u32 = 1u32;
const SHADE_CONFIG_HAS_COLOR: u32 = 2u32;

const SHADE_SHADER: &str = include_str!("shaders/shade.wgsl");

fn shade_shader() -> String {
    SHADE_SHADER.to_owned() + super::PACKED_VOXEL + shaders::COMMON
}

tag!(
    pub ShadedImageTag, u32, VoxelSize, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU shaded (RGBA) images"
);

/// Handle to configuration buffers used when computing shading
pub type Workspace = ConfigWorkspace<ShadeConfig>;

/// Handle to size-dependent buffers used when computing shading
pub struct Buffers {
    pub(crate) out: FlexBuffer<ShadedImageTag>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) has_color: bool,
}

impl Buffers {
    pub(crate) fn new(
        gpu: &Gpu,
        image_size: VoxelSize,
    ) -> Result<Buffers, BufferSizeError> {
        let out = FlexBuffer::new(&gpu.device, "shade output", image_size)?;
        let bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge output bind group"),
                layout: &gpu.output_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.bind_active(),
                }],
            });
        Ok(Buffers {
            out,
            bind_group,
            has_color: false,
        })
    }

    /// Returns a reference to the output buffer
    pub fn output(&self) -> &FlexBuffer<ShadedImageTag> {
        &self.out
    }
}

/// Error returned when submitting a shade operation
#[derive(Debug, thiserror::Error)]
pub enum ShadeError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(VoxelSizeMismatch),

    /// The image size is mismatched between input and SSAO buffers
    #[error(transparent)]
    SsaoSizeMismatch(ImageSizeMismatch),
}

/// Context for building a new shaded image (with optional SSAO)
pub struct Context {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
}

impl crate::Context for Context {
    type Size = VoxelSize;
    type BuffersError = BufferSizeError;
    type Buffers = Buffers;
    type Workspace = Workspace;
    type EncodeError = ShadeError;

    type Args<'a> = &'a super::merge::Buffers;
    type Settings<'a> = Option<&'a super::ssao::Buffers>;

    fn new(gpu: &Gpu) -> Self {
        let shader_code = shade_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects shade pipeline"),
                bind_group_layouts: &[
                    Some(&gpu.config_bind_group_layout),
                    Some(&gpu.input_bind_group_layout), // image
                    Some(&gpu.input_bind_group_layout), // occlusion
                    Some(&gpu.output_bind_group_layout), // out
                ],
                immediate_size: 0u32,
            },
        );
        let shader_module =
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("effects shade shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("effects shade compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("shade_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Self {
            gpu: gpu.clone(),
            pipeline,
        }
    }

    fn buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<Buffers, BufferSizeError> {
        Buffers::new(&self.gpu, image_size)
    }

    fn encode(
        &self,
        image: &super::merge::Buffers,
        ssao: Option<&super::ssao::Buffers>,
        workspace: &mut Workspace,
        buf: &mut Buffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), ShadeError> {
        let size = image.out.size();
        if size != buf.out.size() {
            return Err(ShadeError::ImageSizeMismatch(VoxelSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }));
        }
        if let Some(ssao) = ssao
            && ssao.blurred_occlusion.size() != size.into()
        {
            return Err(ShadeError::SsaoSizeMismatch(ImageSizeMismatch {
                expected: size.into(),
                actual: ssao.blurred_occlusion.size(),
            }));
        }

        // Load the config before beginning the compute pass
        let cfg = ShadeConfig {
            image_size: [size.width(), size.height(), size.depth()],
            flags: if ssao.is_some() {
                SHADE_CONFIG_HAS_SSAO
            } else {
                0
            } | if buf.has_color {
                SHADE_CONFIG_HAS_COLOR
            } else {
                0
            },
        };
        workspace.encode_copy(encoder, staging, &cfg);

        // Scope to bound the lifetime of compute_pass
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shade compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, &image.ro_bind_group, &[]);
        compute_pass.set_bind_group(
            2,
            ssao.map(|s| &s.output_bind_group)
                .unwrap_or(&image.ro_bind_group), // dummy binding
            &[],
        );
        compute_pass.set_bind_group(3, &buf.bind_group, &[]);

        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.has_color = false;
        Ok(())
    }

    fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("shade")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_shade_shader() {
        crate::compile_shader(&shade_shader(), "shade");
    }

    #[test]
    fn shade_config_layout() {
        crate::test::compare_struct_layout::<ShadeConfig>(
            &shade_shader(),
            "ShadeConfig",
        );
    }
}
