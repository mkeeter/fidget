//! Image merging post-processing pass
//!
//! This pass starts with [`GeometryPixel`](fidget_raster::voxel::GeometryPixel)
//! buffers (16 bytes per pixel, stored on the GPU).  It merges and denoises
//! these buffers into a single image containing [`PackedVoxel`] data (normals,
//! depth, and source image index packed into 8 bytes per pixel).
use crate::{
    ConfigWorkspace, Gpu, VoxelSizeMismatch,
    buf::{BufferSizeError, FlexBuffer},
    shaders, tag,
    voxel::PackedVoxel,
};
use fidget_core::render::VoxelSize;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

fn merge_shader() -> String {
    MERGE_SHADER.to_owned() + super::PACKED_VOXEL + shaders::COMMON
}

/// GPU-side configuration object for merging images
///
/// This is public because it's used in a public type, but is not expected to be
/// used directly.
#[derive(Copy, Clone, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[cfg_attr(test, derive(facet::Facet))]
#[repr(C)]
#[doc(hidden)]
pub struct MergeConfig {
    /// Image size, in pixels
    pub image_size: [u32; 2],

    /// Whether or not to denoise when merging (non-zero is true)
    pub denoise: u32,

    /// Scale to compensate for z-flattening
    pub z_scale: f32,

    /// Offset applied to indices when merging
    ///
    /// When this is 0, we initialize the output image
    pub index_base: u32,

    // Padding to next multiple of 8 bytes
    _pad: u32,
}

/// Settings for image merging pass
#[derive(Copy, Clone, Debug)]
pub struct Settings {
    /// Denoise normals (replacing backwards-facing normals with a neighbor)
    pub denoise: bool,

    /// Non-uniform Z scaling applied to the model
    ///
    /// The Z normal is *divided* by this value when merging images, to cancel
    /// out the effect of the scaling.
    pub z_scale: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            denoise: true,
            z_scale: 1.0,
        }
    }
}

tag!(pub MergeVoxelBufferTag, PackedVoxel, VoxelSize, STORAGE | COPY_SRC,
    "Buffer tag for on-GPU merged ([`PackedVoxel`]) images"
);

/// Handle to workspace buffers used when merging images
pub type Workspace = ConfigWorkspace<MergeConfig>;

/// Handle to size-dependent buffers used when merging images
pub struct Buffers {
    pub(crate) out: FlexBuffer<MergeVoxelBufferTag>,
    /// Bind group holding `out` as a read-only value
    pub(crate) ro_bind_group: wgpu::BindGroup,
    /// Bind group holding `out` as a read-write value
    pub(crate) rw_bind_group: wgpu::BindGroup,
    /// Stateful tracker for number of images accumulated
    pub(crate) image_count: usize,
}

impl Buffers {
    pub(crate) fn new(
        gpu: &Gpu,
        image_size: VoxelSize,
    ) -> Result<Self, BufferSizeError> {
        let out = FlexBuffer::new(&gpu.device, "merge output", image_size)?;
        let rw_bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge output bind group"),
                layout: &gpu.output_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.bind_active(),
                }],
            });
        let ro_bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge output bind group"),
                layout: &gpu.input_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.bind_active(),
                }],
            });
        Ok(Buffers {
            out,
            rw_bind_group,
            ro_bind_group,
            image_count: 0,
        })
    }

    /// Returns a handle to the output buffer
    pub fn output(&self) -> &FlexBuffer<MergeVoxelBufferTag> {
        &self.out
    }

    /// Resets the merge buffer
    ///
    /// The next call to [`Context::submit`] will clear the buffer and
    /// begin accumulating from scratch.
    pub fn reset(&mut self) {
        self.image_count = 0;
    }
}

/// Error returned when submitting a merge operation
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// The image size is mismatched between input and output buffers
    #[error(transparent)]
    ImageSizeMismatch(#[from] VoxelSizeMismatch),
}

/// WGPU context for merging images
///
/// Note that [`Buffers::reset`] should be called before the first call to
/// `submit` or `encode`.  The input image size must match the size of the
/// [`Buffers`]; otherwise an error will be returned.
pub struct Context {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
}

impl crate::Context for Context {
    type Workspace = Workspace;
    type Buffers = Buffers;
    type BuffersError = BufferSizeError;
    type Size = VoxelSize;
    type Args<'a> = &'a super::render::Buffers;
    type Settings<'a> = Settings;
    type EncodeError = MergeError;

    /// Builds a new context for applying effects
    fn new(gpu: &Gpu) -> Self {
        let shader_code = merge_shader();
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("effects merge pipeline"),
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
                    label: Some("effects merge shader module"),
                    source: wgpu::ShaderSource::Wgsl(shader_code.into()),
                });
        let pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("voxel effects merge compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("merge_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        Self {
            gpu: gpu.clone(),
            pipeline,
        }
    }

    /// Builds a new [`Workspace`]
    fn workspace(&self) -> Workspace {
        self.gpu.config_workspace("merge")
    }

    /// Builds a new set of [`Buffers`] for a particular render size
    fn buffers(
        &self,
        image_size: VoxelSize,
    ) -> Result<Buffers, BufferSizeError> {
        Buffers::new(&self.gpu, image_size)
    }

    fn encode(
        &self,
        image: &super::render::Buffers,
        settings: Settings,
        workspace: &mut Workspace,
        buf: &mut Buffers,
        encoder: &mut wgpu::CommandEncoder,
        staging: &mut wgpu::util::StagingBelt,
    ) -> Result<(), MergeError> {
        let size = image.output().size();
        if buf.out.size() != size {
            return Err(VoxelSizeMismatch {
                expected: size,
                actual: buf.out.size(),
            }
            .into());
        }
        let cfg = MergeConfig {
            image_size: [size.width(), size.height()],
            denoise: settings.denoise as u32,
            index_base: buf.image_count as u32,
            z_scale: settings.z_scale,
            _pad: 0,
        };
        let mut writer = staging.write_buffer(
            encoder,
            &workspace.config,
            0,
            (std::mem::size_of::<MergeConfig>() as u64)
                .try_into()
                .unwrap(),
        );
        writer.copy_from_slice(cfg.as_bytes());

        // Scope to bound the lifetime of compute_pass
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge compute pass"),
                timestamp_writes: None, // TODO add timestamps?
            });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &workspace.bind_group, &[]);
        compute_pass.set_bind_group(1, image.output_bind_group(), &[]);
        compute_pass.set_bind_group(2, &buf.rw_bind_group, &[]);
        compute_pass.dispatch_workgroups(
            size.width().div_ceil(8),
            size.height().div_ceil(8),
            1,
        );
        buf.image_count += 1;
        Ok(())
    }
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
